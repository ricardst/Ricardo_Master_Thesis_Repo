# src/run_autotabpfn_hpo.py

import os
import pandas as pd
import numpy as np
import logging
import torch
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

try:
    from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier
    from tabpfn_extensions import interpretability # Added for SHAP
except ImportError:
    print("Error: Required libraries not found for AutoTabPFN.")
    print("Please install tabpfn and tabpfn-extensions:")
    print("  pip install tabpfn")
    print("  pip install git+https://github.com/automl/TabPFN-extensions.git")
    exit(1)
# Assuming utils.py and config_loader.py are in the same src directory or PYTHONPATH is set
try:
    from . import utils
    from .config_loader import load_config
except ImportError:
    # Fallback if run directly
    import utils
    from config_loader import load_config

def prepare_data_for_tabpfn(config, project_root):
    """
    Loads the tabular feature DataFrame and applies filtering and selection
    based on the configuration, inspired by the TCN HPO preparation script.
    
    Returns:
        - X (np.ndarray): Feature matrix.
        - y (np.ndarray): Numerically encoded labels.
        - subject_ids (np.ndarray): Corresponding subject IDs for each window.
        - label_encoder (LabelEncoder): The fitted encoder object to decode predictions.
    """
    logging.info("--- Starting Data Preparation for AutoTabPFN ---")

    intermediate_dir = os.path.join(project_root, config.get('intermediate_feature_dir', 'features'))
    tabular_data_path = os.path.join(intermediate_dir, "tabular_features_for_tabpfn.pkl")

    if not os.path.exists(tabular_data_path):
        logging.error(f"Required data file not found: {tabular_data_path}")
        return None, None, None, None

    try:
        df = utils.load_pickle(tabular_data_path)
        logging.info(f"Successfully loaded tabular data. Initial shape: {df.shape}")
    except Exception as e:
        logging.error(f"Error loading data from {tabular_data_path}: {e}")
        return None, None, None, None

    target_col = config.get('target_column', 'Activity')
    subject_col = config.get('subject_id_column', 'SubjectID')

    # --- LABEL REMAPPING ---
    logging.info("Attempting to remap activity labels using 'Activity_Mapping_v2.csv'...")
    activity_mapping_path = os.path.join(project_root, "Activity_Mapping_v2.csv")
    if os.path.exists(activity_mapping_path):
        try:
            mapping_df = pd.read_csv(activity_mapping_path)
            if 'Former_Label' in mapping_df.columns and 'New_Label' in mapping_df.columns:
                label_mapping_dict = pd.Series(mapping_df.New_Label.values, index=mapping_df.Former_Label).to_dict()
                original_labels = df[target_col].copy()
                df[target_col] = df[target_col].map(label_mapping_dict).fillna(original_labels)
                changed_count = (original_labels != df[target_col]).sum()
                logging.info(f"Successfully remapped {changed_count} labels out of {len(df)}.")
            else:
                logging.warning("Activity_Mapping_v2.csv missing 'Former_Label' or 'New_Label' columns. Skipping.")
        except Exception as e:
            logging.error(f"Error processing Activity_Mapping_v2.csv: {e}. Skipping remapping.")
    else:
        logging.warning(f"Activity_Mapping_v2.csv not found at {activity_mapping_path}. Skipping remapping.")

    # --- MANUAL SUBJECT EXCLUSION ---
    excluded_subjects = config.get('excluded_subjects_manual', ['OutSense-036', 'OutSense-425', 'OutSense-515'])
    if excluded_subjects:
        initial_count = len(df)
        df = df[~df[subject_col].isin(excluded_subjects)]
        logging.info(f"Manually excluded {initial_count - len(df)} windows from subjects: {excluded_subjects}.")
        if df.empty:
            logging.error("All data removed after manual subject exclusion."); return None, None, None, None

    # --- UNCONDITIONAL CLASS REMOVAL ---
    classes_to_remove = config.get('unconditionally_removed_classes', ['Other', 'Unknown'])
    if classes_to_remove:
        initial_count = len(df)
        df = df[~df[target_col].isin(classes_to_remove)]
        logging.info(f"Removed {initial_count - len(df)} windows from classes: {classes_to_remove}.")
        if df.empty:
            logging.error(f"All data removed after excluding {classes_to_remove}."); return None, None, None, None

    # --- FILTER FOR SELECTED CLASSES ---
    selected_classes = config.get('selected_classes', ['Propulsion', 'Resting', 'Eating'])
    if selected_classes:
        initial_count = len(df)
        df = df[df[target_col].isin(selected_classes)]
        logging.info(f"Filtered for 'selected_classes': Kept {len(df)} from {initial_count} windows for {selected_classes}.")
        if df.empty:
            logging.error("All data removed after 'selected_classes' filtering."); return None, None, None, None

    # --- REMOVE CLASSES WITH FEW INSTANCES ---
    min_instances = config.get('min_class_instances', 10)
    if min_instances > 0:
        class_counts = df[target_col].value_counts()
        labels_to_keep = class_counts[class_counts >= min_instances].index.tolist()
        if len(labels_to_keep) < len(class_counts):
            initial_count = len(df)
            df = df[df[target_col].isin(labels_to_keep)]
            logging.info(f"Removed classes with < {min_instances} instances. Kept {len(df)} from {initial_count} windows.")
            if df.empty:
                logging.error("All data removed after min_class_instances filtering."); return None, None, None, None
        else:
            logging.info("All remaining classes meet the minimum instance threshold.")

    logging.info(f"Final data shape after all filtering: {df.shape}")
    logging.info(f"Final class distribution:\n{df[target_col].value_counts()}")

    # --- FINAL DATA PREPARATION ---
    # Define feature columns (all columns except identifiers and target)
    feature_cols = [col for col in df.columns if col not in [target_col, subject_col, 'window_start_time', 'window_end_time']]
    
    # Limit features to 500 if there are more
    max_features = 100
    if len(feature_cols) > max_features:
        logging.info(f"Found {len(feature_cols)} features. Limiting to first {max_features} features.")
        feature_cols = feature_cols[:max_features]
        logging.info(f"Using {len(feature_cols)} features after truncation.")
    else:
        logging.info(f"Using all {len(feature_cols)} features (within limit of {max_features}).")
    
    # --- DATA VALIDATION AND CLEANING ---
    logging.info("Starting data validation and cleaning...")
    
    # Extract feature data for cleaning
    X_df = df[feature_cols].copy()
    initial_shape = X_df.shape
    logging.info(f"Initial feature matrix shape: {initial_shape}")
    
    # Check for infinity values
    inf_mask = np.isinf(X_df.values)
    if np.any(inf_mask):
        inf_count = np.sum(inf_mask)
        logging.warning(f"Found {inf_count} infinity values. Replacing with NaN for proper handling.")
        X_df = X_df.replace([np.inf, -np.inf], np.nan)
    
    # Check for NaN values
    nan_mask = X_df.isnull()
    if nan_mask.any().any():
        nan_count = nan_mask.sum().sum()
        logging.warning(f"Found {nan_count} NaN values. Handling NaN values...")
        
        # Option 1: Fill NaN with median (more robust than mean)
        for col in X_df.columns:
            if X_df[col].isnull().any():
                median_val = X_df[col].median()
                if pd.isnull(median_val):  # If all values are NaN
                    logging.warning(f"Column {col} has all NaN values. Filling with 0.")
                    X_df[col].fillna(0, inplace=True)
                else:
                    X_df[col].fillna(median_val, inplace=True)
                    logging.info(f"Filled {nan_mask[col].sum()} NaN values in column {col} with median: {median_val:.6f}")
    
    # Check for extremely large values that might cause overflow
    X_values = X_df.values
    max_abs_value = np.max(np.abs(X_values))
    logging.info(f"Maximum absolute value in features: {max_abs_value:.6e}")
    
    # Cap extremely large values (beyond float32 safe range)
    max_safe_value = 1e30  # Safe threshold for float32
    if max_abs_value > max_safe_value:
        logging.warning(f"Found extremely large values (max: {max_abs_value:.6e}). Capping at ±{max_safe_value:.6e}")
        X_values = np.clip(X_values, -max_safe_value, max_safe_value)
        X_df = pd.DataFrame(X_values, columns=feature_cols, index=X_df.index)
    
    # Convert to float32 and check for any remaining issues
    X = X_df.astype(np.float32).values
    
    # Final validation
    if np.any(np.isnan(X)):
        logging.error("NaN values still present after cleaning!")
        return None, None, None, None
    
    if np.any(np.isinf(X)):
        logging.error("Infinity values still present after cleaning!")
        return None, None, None, None
    
    # Check for finite values
    if not np.all(np.isfinite(X)):
        logging.error("Non-finite values detected after cleaning!")
        return None, None, None, None
    
    logging.info(f"Data cleaning complete. Final feature matrix shape: {X.shape}")
    logging.info(f"Feature value range: [{np.min(X):.6f}, {np.max(X):.6f}]")
    
    y_str = df[target_col].values
    subject_ids = df[subject_col].values

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_str)
    
    logging.info(f"Label encoding complete. Found {len(label_encoder.classes_)} classes.")
    logging.info("Data preparation finished.")
    
    return X, y_encoded, subject_ids, label_encoder


def run_autotabpfn_hpo(config, project_root):
    """
    Main function to run the AutoTabPFN training and evaluation pipeline.
    """
    logging.info("========= Starting AutoTabPFN HPO Pipeline =========")
    
    # 1. Prepare Data
    X, y, subjects, label_encoder = prepare_data_for_tabpfn(config, project_root)
    if X is None:
        logging.error("Data preparation failed. Exiting pipeline.")
        return

    # 1.5. Limit samples to 10,000 if there are more
    max_samples = 10000
    if len(X) > max_samples:
        logging.info(f"Dataset has {len(X)} samples. Limiting to {max_samples} samples through random sampling.")
        
        # Use stratified sampling to maintain class distribution
        from sklearn.model_selection import train_test_split
        random_state = config.get('seed_number', 42)
        
        # Sample indices while maintaining class distribution
        _, sample_indices = train_test_split(
            np.arange(len(X)), 
            test_size=max_samples, 
            random_state=random_state,
            stratify=y
        )
        
        # Apply sampling
        X = X[sample_indices]
        y = y[sample_indices]
        subjects = subjects[sample_indices]
        
        logging.info(f"Sampled dataset to {len(X)} samples. New class distribution:")
        unique_classes, class_counts = np.unique(y, return_counts=True)
        for cls, count in zip(unique_classes, class_counts):
            class_name = label_encoder.classes_[cls]
            logging.info(f"  {class_name}: {count} samples")
    else:
        logging.info(f"Dataset has {len(X)} samples (within limit of {max_samples}). No sampling needed.")

    # 2. Split Data - Subject-based splitting
    hpo_config = config.get('autotabpfn_hpo_config', {})
    test_size = hpo_config.get('test_size', 0.2)
    random_state = config.get('seed_number', 42)

    # Check if specific test subjects are defined in config
    test_subjects = config.get('test_subjects', [])
    
    if test_subjects:
        logging.info(f"Using predefined test subjects from config: {test_subjects}")
        test_mask = np.isin(subjects, test_subjects)
        train_mask = ~test_mask
        
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        train_subjects_list = np.unique(subjects[train_mask])
        test_subjects_list = np.unique(subjects[test_mask])
        
        if len(X_test) == 0:
            logging.warning("No data for the test set based on provided test_subjects. Falling back to GroupShuffleSplit.")
            test_subjects = []  # Trigger fallback
    
    if not test_subjects:
        logging.info("Using GroupShuffleSplit for subject-based train/test splitting")
        # Use GroupShuffleSplit to ensure subject-level separation
        unique_subjects = np.unique(subjects)
        
        if len(unique_subjects) < 2:
            logging.warning("Only one subject found. Using random split instead of subject-based split.")
            train_idx, test_idx = train_test_split(
                np.arange(len(X)), test_size=test_size, random_state=random_state, 
                stratify=y
            )
        else:
            gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            train_idx, test_idx = next(gss.split(X, y, subjects))
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        train_subjects_list = np.unique(subjects[train_idx])
        test_subjects_list = np.unique(subjects[test_idx])

    logging.info(f"Data split using {len(train_subjects_list)} training subjects and {len(test_subjects_list)} test subjects.")
    logging.info(f"Train set shape: X={X_train.shape}, y={y_train.shape}")
    logging.info(f"Test set shape:  X={X_test.shape}, y={y_test.shape}")
    
    # 2.5. Feature Scaling for numerical stability
    logging.info("Applying feature scaling for numerical stability...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Validate scaled data
    if not np.all(np.isfinite(X_train_scaled)):
        logging.error("Non-finite values in scaled training data!")
        return
    if not np.all(np.isfinite(X_test_scaled)):
        logging.error("Non-finite values in scaled test data!")
        return
    
    logging.info(f"Feature scaling complete. Train range: [{np.min(X_train_scaled):.6f}, {np.max(X_train_scaled):.6f}]")
    logging.info(f"Test range: [{np.min(X_test_scaled):.6f}, {np.max(X_test_scaled):.6f}]")
    
    # 3. Configure and Train Model
    # GPU and Job configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_jobs = 1 # Get n_jobs from config
    logging.info(f"Device set to: '{device}'.")
    # Note: AutoTabPFN leverages the GPU for parallelism. The n_jobs parameter is noted for configuration
    # but does not directly apply to the AutoTabPFN trainer itself, unlike sklearn models.
    logging.info(f"Parallel jobs (n_jobs) configured to: {n_jobs}. Parallelism is primarily handled by the GPU.")

    n_ensemble_configurations = hpo_config.get('n_ensemble_configurations', 32)
    
    logging.info(f"Initializing AutoTabPFN with {n_ensemble_configurations} ensemble configurations.")
    classifier = AutoTabPFNClassifier(
        max_time=10800,
        device=device,
        ignore_pretraining_limits=True
    )

    logging.info("Starting model training...")
    classifier.fit(X_train_scaled, y_train)
    logging.info("Model training complete.")

    # 4. Evaluate Model
    logging.info("Evaluating model on the test set...")
    y_pred = classifier.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, digits=3)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    logging.info(f"Test Set Accuracy: {accuracy:.4f}")
    logging.info("Classification Report:\n" + report)

    # 5. Save Results
    output_dir = os.path.join(project_root, config.get('results_dir', 'results'), 'tabpfn_hpo')
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Saving results to: {output_dir}")

    # Save metrics report with "tabpfn" in the name
    with open(os.path.join(output_dir, 'tabpfn_classification_report.txt'), 'w') as f:
        f.write(f"Test Set Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    # Save confusion matrix plot with "tabpfn" in the name
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title('TabPFN Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tabpfn_confusion_matrix.png'))
    plt.close()
    logging.info("Saved TabPFN classification report and confusion matrix plot.")

    # Save model and label encoder with "tabpfn" in their names
    model_path = os.path.join(output_dir, 'tabpfn_model.joblib')
    encoder_path = os.path.join(output_dir, 'tabpfn_label_encoder.joblib')
    joblib.dump(classifier, model_path)
    joblib.dump(label_encoder, encoder_path)
    logging.info(f"Saved trained TabPFN model to {model_path} and label encoder to {encoder_path}.")

    # Save scaler along with model
    scaler_path = os.path.join(output_dir, 'tabpfn_scaler.joblib')
    joblib.dump(scaler, scaler_path)
    logging.info(f"Saved feature scaler to {scaler_path}.")

    logging.info("========= AutoTabPFN HPO Pipeline Finished =========")


if __name__ == '__main__':
    # Configure logging to write to both console and file
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler('autotabpfn_logs', mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(file_handler)
    
    logging.info("Logging configured to write to console and 'autotabpfn_logs' file")

    try:
        script_dir = os.path.dirname(__file__)
        project_root = os.path.abspath(os.path.join(script_dir, '..'))
        config_file_path = os.path.join(project_root, 'config.yaml')
        cfg = load_config(config_file_path)

        utils.set_seed(cfg.get('seed_number', 42))

        run_autotabpfn_hpo(cfg, project_root)

    except FileNotFoundError as e:
        logging.error(f"Configuration or data file not found: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred in the main execution block: {e}", exc_info=True)