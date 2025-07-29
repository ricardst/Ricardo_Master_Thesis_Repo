# src/feature_selector.py

import os
import logging
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from PyImpetus import PPIMBC # Assuming PyImpetus is installed

# Assuming utils.py is in the same src directory or PYTHONPATH is set
try:
    from . import utils
except ImportError:
    import utils # Fallback if run directly

def get_pyimpetus_model_instance(config, random_state):
    """Instantiates the sklearn model defined in config for PyImpetus."""
    model_type = config.get('pyimpetus_model_type', 'RandomForestClassifier')
    model_params = config.get('pyimpetus_model_params', {})

    # Ensure random_state is passed if the model accepts it
    if 'random_state' in model_params:
         model_params['random_state'] = random_state

    # Set n_jobs=1 for underlying model as PPIMBC handles parallelism
    if 'n_jobs' in model_params:
        logging.debug(f"Overriding n_jobs to 1 for PyImpetus underlying model {model_type}")
        model_params['n_jobs'] = 1

    logging.info(f"Initializing PyImpetus underlying model: {model_type} with params: {model_params}")

    try:
        if model_type == 'RandomForestClassifier':
            model_params.setdefault('class_weight', 'balanced') # Good default
            return RandomForestClassifier(**model_params)
        elif model_type == 'DecisionTreeClassifier':
             model_params.setdefault('class_weight', 'balanced') # Good default
             return DecisionTreeClassifier(**model_params)
        elif model_type == 'LogisticRegression':
             model_params.setdefault('class_weight', 'balanced') # Good default
             model_params.setdefault('solver', 'liblinear') # Often needed
             return LogisticRegression(**model_params)
        elif model_type == 'SVC':
             model_params.setdefault('class_weight', 'balanced') # Good default
             model_params.setdefault('probability', True) # Needed by some internal metrics? Check PyImpetus docs if issues
             return SVC(**model_params)
        elif model_type == 'GradientBoostingClassifier':
             return GradientBoostingClassifier(**model_params)
        # Add other models as needed
        else:
            logging.warning(f"Unsupported pyimpetus_model_type: '{model_type}'. Defaulting to DecisionTreeClassifier.")
            return DecisionTreeClassifier(random_state=random_state, class_weight='balanced')
    except Exception as e:
         logging.error(f"Error initializing model {model_type}: {e}. Defaulting to DecisionTreeClassifier.", exc_info=True)
         return DecisionTreeClassifier(random_state=random_state, class_weight='balanced')


def run_feature_selection(engineered_features_path,
                          y_windows_path,
                          subject_ids_path,
                          engineered_names_path,
                          config):
    """
    Runs PyImpetus feature selection on the engineered features.

    Args:
        engineered_features_path (str): Path to the .npy file with engineered features.
        y_windows_path (str): Path to the .npy file with window labels.
        subject_ids_path (str): Path to the .npy file with window subject IDs.
        engineered_names_path (str): Path to the .pkl file with engineered feature names.
        config (dict): The configuration dictionary.

    Returns:
        str: Path to the saved file containing the list of selected feature names.
             Returns None if feature selection fails or produces no features.
    """
    logging.info("--- Running PyImpetus Markov Blanket Feature Selection ---")

    # Load necessary data
    try:
        logging.info(f"Loading engineered features from: {engineered_features_path}")
        engineered_features = utils.load_numpy(engineered_features_path)
        logging.info(f"Loading window labels from: {y_windows_path}")
        y_windows = utils.load_numpy(y_windows_path)
        logging.info(f"Loading subject IDs from: {subject_ids_path}")
        subject_ids_windows = utils.load_numpy(subject_ids_path, allow_pickle=True) # Allow object dtype for IDs
        logging.info(f"Loading engineered feature names from: {engineered_names_path}")
        engineered_feature_names = utils.load_pickle(engineered_names_path)
    except FileNotFoundError as e:
        logging.error(f"Input file not found for feature selection: {e}. Aborting selection.")
        return None
    except Exception as e:
        logging.error(f"Error loading input files for feature selection: {e}", exc_info=True)
        return None

    if not isinstance(engineered_feature_names, list) or len(engineered_feature_names) != engineered_features.shape[1]:
        logging.error("Mismatch between loaded engineered features and feature names. Aborting.")
        logging.error(f"Features shape: {engineered_features.shape}, Number of names: {len(engineered_feature_names)}")
        return None

    if engineered_features.shape[1] == 0:
        logging.error("Cannot run feature selection: Input engineered features array has no columns.")
        return None
    if engineered_features.shape[0] != y_windows.shape[0] or engineered_features.shape[0] != subject_ids_windows.shape[0]:
        logging.error("Mismatch in number of samples between features, labels, and subject IDs. Aborting.")
        return None


    # Split data to get the training set for PyImpetus fitting
    test_subjects = config.get('test_subjects', [])
    if not test_subjects:
        logging.warning("No test subjects defined in config. Feature selection will run on all data.")
        train_mask = np.ones(len(subject_ids_windows), dtype=bool)
    else:
        logging.info(f"Using training data (excluding subjects: {test_subjects}) for feature selection.")
        test_mask = np.isin(subject_ids_windows, test_subjects)
        train_mask = ~test_mask

    engineered_features_train_fs = engineered_features[train_mask]
    y_windows_train_fs = y_windows[train_mask]

    if engineered_features_train_fs.shape[0] == 0:
         logging.error("Cannot run feature selection: No training samples available after splitting.")
         return None

    # Encode labels
    temp_label_encoder = LabelEncoder()
    try:
        y_train_encoded_fs = temp_label_encoder.fit_transform(y_windows_train_fs)
    except Exception as e:
        logging.error(f"Error encoding labels for feature selection: {e}", exc_info=True)
        return None

    # Create DataFrame for PyImpetus
    engineered_features_df_train = pd.DataFrame(
        engineered_features_train_fs,
        columns=engineered_feature_names
    )

    logging.info(f"Running PPIMBC with {engineered_features_df_train.shape[1]} features on {engineered_features_df_train.shape[0]} training samples.")
    seed = config.get('seed_number', 42)
    pyimp_cv = config.get('pyimpetus_cv', 0)
    pyimp_num_simul = config.get('pyimpetus_num_simul', 10)
    pyimp_pval = config.get('pyimpetus_p_val_thresh', 0.05)
    logging.info(f"PyImpetus params: num_simul={pyimp_num_simul}, p_val={pyimp_pval}, cv={pyimp_cv}")

    # Initialize PPIMBC model
    underlying_model = get_pyimpetus_model_instance(config, seed)
    ppmbc_model = PPIMBC(
        model=underlying_model,
        p_val_thresh=pyimp_pval,
        num_simul=pyimp_num_simul,
        simul_size=config.get('pyimpetus_simul_size', 0.2),
        simul_type=0, # 0=non-stratified, 1=stratified
        sig_test_type="non-parametric",
        cv=pyimp_cv,
        verbose=2,
        random_state=seed,
        n_jobs=1, # Use all cores for PPIMBC's internal parallelism
    )

    selected_feature_names = None
    try:
        # Fit the model (this is the time-consuming part)
        logging.info("Fitting PyImpetus PPIMBC model...")
        ppmbc_model.fit(engineered_features_df_train, y_train_encoded_fs)
        # Get the selected Markov Blanket features
        selected_feature_names = ppmbc_model.MB

        if not selected_feature_names:
            logging.warning("PyImpetus did not select any features (Markov Blanket is empty).")
            selected_feature_names = [] # Ensure it's an empty list
        else:
             logging.info(f"PyImpetus selected {len(selected_feature_names)} features:")
             logging.info(f"{selected_feature_names}")

    except Exception as e:
        logging.error(f"Error during PyImpetus execution: {e}", exc_info=True)
        return None # Return None if FS failed

    # --- Save the selected feature names ---
    results_dir = config.get('results_dir', 'results')
    output_filename = config.get('feature_selection_output_file', 'selected_features_pyimpetus.pkl')
    output_path = os.path.join(results_dir, output_filename)

    try:
        utils.save_pickle(selected_feature_names, output_path)
    except Exception as e:
        logging.error(f"Failed to save selected features to {output_path}. Error: {e}")
        return None # Indicate failure if saving failed

    # Optional: Plot feature importance
    if selected_feature_names: # Only plot if features were selected
        try:
            logging.info("Generating feature importance plot...")
            fig = ppmbc_model.feature_importance()
            if fig:
                 plot_path = os.path.join(results_dir, 'pyimpetus_feature_importance.png')
                 fig.savefig(plot_path)
                 logging.info(f"Saved PyImpetus feature importance plot to '{plot_path}'")
                 plt.close(fig) # Close the plot
        except Exception as plot_e:
            logging.warning(f"Could not generate/save feature importance plot: {plot_e}")

    return output_path # Return path to the saved list

if __name__ == '__main__':
    # Example of how to run this script directly
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        # Assumes config is loaded from ../config.yaml relative to src/
        # Assumes feature_engineering.py has run and produced outputs in features/
        from config_loader import load_config
        script_dir = os.path.dirname(__file__)
        project_root = os.path.abspath(os.path.join(script_dir, '..'))
        config_file_path = os.path.join(project_root, 'config.yaml')
        cfg = load_config(config_file_path)

        utils.set_seed(cfg.get('seed_number', 42))

        # Define input paths based on assumed output of feature_engineering stage
        feature_dir = os.path.join(project_root, cfg.get('intermediate_feature_dir', 'features'))
        eng_feat_path = os.path.join(feature_dir, 'engineered_features.npy')
        y_win_path = os.path.join(feature_dir, 'y_windows.npy')
        subj_ids_path = os.path.join(feature_dir, 'subject_ids_windows.npy')
        eng_names_path = os.path.join(feature_dir, 'engineered_feature_names.pkl')

        # Check if required input files exist
        required_files = [eng_feat_path, y_win_path, subj_ids_path, eng_names_path]
        if not all(os.path.exists(p) for p in required_files):
             logging.error("One or more input files for feature selection not found. Please run feature_engineering stage first.")
             missing = [p for p in required_files if not os.path.exists(p)]
             logging.error(f"Missing files: {missing}")
        else:
            # Execute the main function of this module
            saved_features_path = run_feature_selection(
                eng_feat_path,
                y_win_path,
                subj_ids_path,
                eng_names_path,
                cfg
            )
            if saved_features_path:
                 logging.info(f"Feature selection complete. Selected features saved to: {saved_features_path}")
            else:
                 logging.error("Feature selection failed.")

    except FileNotFoundError as e:
        logging.error(f"File Not Found Error: {e}")
    except ValueError as e:
        logging.error(f"Value Error: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)