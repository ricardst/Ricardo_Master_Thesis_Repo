# src/data_preparation.py

import os
import logging
import pickle
import numpy as np
import pandas as pd # Only needed if creating DataFrames temporarily
import torch
from sklearn.model_selection import GroupKFold

try:
    from . import utils
    from .utils import SensorDataset # <<< MODIFIED: Import SensorDataset from utils
except ImportError:
    import utils
    from utils import SensorDataset # <<< MODIFIED: Import SensorDataset from utils

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GroupKFold # Not strictly needed, splitting uses masks

# Assuming utils.py is in the same src directory or PYTHONPATH is set
try:
    from . import utils
except ImportError:
    import utils # Fallback if run directly


def run_data_preparation(config,
                         x_windows_raw_path,
                         engineered_features_path,
                         y_windows_path,
                         subject_ids_path,
                         original_feature_names_path,
                         engineered_names_path,
                         selected_features_path=None):
    """
    Loads features, applies selection, combines, splits, scales, encodes,
    and saves prepared data arrays and processors.

    Args:
        config (dict): Configuration dictionary.
        x_windows_raw_path (str): Path to raw feature windows (.npy).
        engineered_features_path (str): Path to engineered features (.npy).
        y_windows_path (str): Path to window labels (.npy).
        subject_ids_path (str): Path to window subject IDs (.npy).
        original_feature_names_path (str): Path to list of original feature names (.pkl).
        engineered_names_path (str): Path to list of engineered feature names (.pkl).
        selected_features_path (str, optional): Path to list of selected eng. feature names (.pkl). Defaults to None.

    Returns:
        dict: Paths to saved artifacts (scaled data arrays, scaler, encoder, etc.).
    """
    logging.info("--- Starting Data Preparation Stage ---")

    # --- Load Data ---
    try:
        logging.info("Loading base data for preparation...")
        X_windows_raw = utils.load_numpy(x_windows_raw_path)
        y_windows = utils.load_numpy(y_windows_path, allow_pickle=True)
        subject_ids_windows = utils.load_numpy(subject_ids_path, allow_pickle=True)
        original_feature_names = utils.load_pickle(original_feature_names_path)
    except Exception as e:
        logging.error(f"Error loading base input files for data preparation: {e}", exc_info=True)
        raise

    n_windows_raw, window_size, n_original_features = X_windows_raw.shape
    logging.info(f"Loaded raw windows shape: {X_windows_raw.shape}")
    if n_original_features != len(original_feature_names):
        logging.warning("Mismatch between raw window features and original names list.")
        # Adjust names or error out? For now, log warning.

    final_engineered_features = None
    final_engineered_names = []

    use_eng_features = config.get('use_engineered_features', True) # Default to True

    if use_eng_features:
        logging.info("Engineered features enabled. Loading and processing...")
        try:
            engineered_features = utils.load_numpy(engineered_features_path)
            engineered_feature_names = utils.load_pickle(engineered_names_path)
            logging.info(f"Loaded engineered features shape: {engineered_features.shape}")

            # Consistency checks
            if engineered_features.shape[0] != n_windows_raw:
                 raise ValueError("Mismatch in window count between raw and engineered features.")
            if engineered_features.shape[1] != len(engineered_feature_names):
                 logging.warning("Mismatch between engineered features dim and names list.")
                 # Adjust names? For now, proceed.

            # Apply Feature Selection (Conditional on config AND use_eng_features)
            final_engineered_features = engineered_features # Start with all loaded
            final_engineered_names = engineered_feature_names
            use_sel_features = config.get('use_selected_features', False)
            sel_feat_path = selected_features_path

            if use_sel_features and sel_feat_path:
                 if os.path.exists(sel_feat_path):
                     try:
                         selected_names = utils.load_pickle(sel_feat_path)
                         logging.info(f"Loaded {len(selected_names)} selected feature names from {sel_feat_path}.")
                         if isinstance(selected_names, list) and engineered_features.shape[1] > 0:
                             # Apply selection
                             name_to_index = {name: idx for idx, name in enumerate(engineered_feature_names)}
                             selected_indices = [name_to_index[name] for name in selected_names if name in name_to_index]
                             # ... (handle missing features, empty indices) ...
                             if selected_indices:
                                 final_engineered_features = engineered_features[:, selected_indices]
                                 final_engineered_names = [engineered_feature_names[i] for i in selected_indices]
                                 logging.info(f"Applied selection: Using {len(final_engineered_names)} engineered features.")
                             else: logging.warning("No valid selected features found. Using all loaded engineered features.")
                         else: logging.warning("Format error in selected features file or no eng features exist. Using all loaded.")
                     except Exception as e: logging.error(f"Error loading/applying selection: {e}. Using all loaded eng features.")
                 else: logging.warning(f"Selected features file not found: {sel_feat_path}. Using all loaded eng features.")
            else: logging.info("Feature selection not enabled or file not provided. Using all loaded eng features.")

        except FileNotFoundError as e:
             logging.error(f"Engineered feature file not found: {e}. Cannot use engineered features.")
             use_eng_features = False # Force disabling if files are missing
             final_engineered_features = None # Ensure it's None
        except Exception as e:
             logging.error(f"Error loading engineered features: {e}. Cannot use engineered features.", exc_info=True)
             use_eng_features = False
             final_engineered_features = None

    else:
        logging.info("Configuration 'use_engineered_features' is False. Skipping engineered features.")
        final_engineered_features = None # Ensure it's None

    # --- Combine Raw and (Selected) Engineered Features ---
    logging.info("Combining features...")
    if use_eng_features and final_engineered_features is not None and final_engineered_features.shape[1] > 0:
        # Combine as before
        n_final_engineered = final_engineered_features.shape[1]
        eng_feat_f32 = final_engineered_features.astype(np.float32)
        x_raw_f32 = X_windows_raw.astype(np.float32)
        eng_feat_reshaped = eng_feat_f32[:, np.newaxis, :]
        eng_feat_repeated = np.repeat(eng_feat_reshaped, window_size, axis=1)
        X_windows_combined = np.concatenate((x_raw_f32, eng_feat_repeated), axis=2)
        final_feature_names = list(original_feature_names) + final_engineered_names
        logging.info(f"Combined raw ({n_original_features}) + engineered ({n_final_engineered}) features.")
    else:
        # Use only raw features
        logging.info("Using only original raw features.")
        X_windows_combined = X_windows_raw.astype(np.float32)
        final_feature_names = list(original_feature_names)
        if use_eng_features and (final_engineered_features is None or final_engineered_features.shape[1] == 0):
            logging.warning("Engineered features were enabled but none were available/selected for combination.")

    logging.info(f"Final combined data shape for splitting: {X_windows_combined.shape}")
    logging.info(f"Final number of features: {len(final_feature_names)}")

    # --- Train/Test Split ---
    logging.info("Performing subject-aware train/test split...")
    test_subjects = config.get('test_subjects', [])
    if not test_subjects:
        logging.error("No 'test_subjects' defined in config for splitting.")
        raise ValueError("Test subjects must be specified in config.")

    test_mask = np.isin(subject_ids_windows, test_subjects)
    train_mask = ~test_mask

    X_train = X_windows_combined[train_mask]
    y_train = y_windows[train_mask]
    train_subject_ids = subject_ids_windows[train_mask] # Save this for CV later

    X_test = X_windows_combined[test_mask]
    y_test = y_windows[test_mask]
    # test_subject_ids = subject_ids_windows[test_mask] # Don't strictly need to save this

    logging.info(f"Split complete. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        logging.error("Train or Test set is empty after splitting.")
        raise ValueError("Empty data split.")

    # --- Encode Labels ---
    logging.info("Encoding labels...")
    label_encoder = LabelEncoder()
    # Fit on combined train/test labels to ensure all classes are seen
    all_labels = np.concatenate((y_train, y_test))
    label_encoder.fit(all_labels)
    y_train_enc = label_encoder.transform(y_train)
    y_test_enc = label_encoder.transform(y_test)
    logging.info(f"Label encoding complete. Classes: {list(label_encoder.classes_)}")

    # --- Scale Features ---
    logging.info("Scaling features (StandardScaler)...")
    scaler = StandardScaler()
    n_train_windows, win_size, n_features = X_train.shape
    input_channels = n_features # This is the actual number of channels in the prepared data
    logging.info(f"Number of features to scale (input_channels): {input_channels}")

    # Reshape for scaler: (N, W, C) -> (N*W, C)
    X_train_reshaped_for_scaling = X_train.reshape(-1, n_features)

    # Clean just before scaling as safeguard
    if np.isnan(X_train_reshaped_for_scaling).any() or np.isinf(X_train_reshaped_for_scaling).any():
        logging.warning("NaNs/Infs found before scaling train data. Applying nan_to_num.")
        X_train_reshaped_for_scaling = np.nan_to_num(X_train_reshaped_for_scaling, nan=0.0, posinf=0.0, neginf=0.0)

    logging.info("Fitting scaler on training data...")
    scaler.fit(X_train_reshaped_for_scaling)

    # Transform train data
    X_train_scaled_flat = scaler.transform(X_train_reshaped_for_scaling)
    # Reshape back to (N, W, C)
    X_train_scaled = X_train_scaled_flat.reshape(n_train_windows, win_size, n_features)

    # Transform test data
    n_test_windows = X_test.shape[0]
    X_test_reshaped_for_scaling = X_test.reshape(-1, n_features)
    if np.isnan(X_test_reshaped_for_scaling).any() or np.isinf(X_test_reshaped_for_scaling).any():
        logging.warning("NaNs/Infs found before scaling test data. Applying nan_to_num.")
        X_test_reshaped_for_scaling = np.nan_to_num(X_test_reshaped_for_scaling, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_scaled_flat = scaler.transform(X_test_reshaped_for_scaling)
    X_test_scaled = X_test_scaled_flat.reshape(n_test_windows, win_size, n_features)
    logging.info("Scaling complete.")

    # --- Reshape for CNN Input ---
    # PyTorch Conv1d expects (N, C, L) = (N, n_features, win_size)
    logging.info("Reshaping data for Conv1D input (N, C, L)...")
    X_train_final = X_train_scaled.transpose(0, 2, 1)
    X_test_final = X_test_scaled.transpose(0, 2, 1)
    logging.info(f"Final train data shape: {X_train_final.shape}")
    logging.info(f"Final test data shape: {X_test_final.shape}")

    # --- Save Outputs ---
    results_dir = config.get('results_dir', 'results')
    prep_data_dir = os.path.join(results_dir, 'prepared_data')
    os.makedirs(prep_data_dir, exist_ok=True)
    output_paths = {}

    try:
        logging.info(f"Saving prepared data arrays and processors to: {prep_data_dir}")
        output_paths['X_train'] = os.path.join(prep_data_dir, 'X_train.npy')
        utils.save_numpy(X_train_final, output_paths['X_train'])
        output_paths['y_train'] = os.path.join(prep_data_dir, 'y_train.npy')
        utils.save_numpy(y_train_enc, output_paths['y_train'])
        output_paths['X_test'] = os.path.join(prep_data_dir, 'X_test.npy')
        utils.save_numpy(X_test_final, output_paths['X_test'])
        output_paths['y_test'] = os.path.join(prep_data_dir, 'y_test.npy')
        utils.save_numpy(y_test_enc, output_paths['y_test'])

        output_paths['scaler'] = os.path.join(prep_data_dir, 'scaler.pkl')
        utils.save_pickle(scaler, output_paths['scaler'])
        output_paths['label_encoder'] = os.path.join(prep_data_dir, 'label_encoder.pkl')
        utils.save_pickle(label_encoder, output_paths['label_encoder'])

        output_paths['train_subject_ids'] = os.path.join(prep_data_dir, 'train_subject_ids.npy')
        utils.save_numpy(train_subject_ids, output_paths['train_subject_ids'])

        # Save summary/metadata
        summary = {
            'input_channels': input_channels, # This now reflects combined or raw count
            'num_original_features': n_original_features, # Explicitly save original count
            'num_engineered_features': final_engineered_features.shape[1] if final_engineered_features is not None else 0,
            'engineered_features_used': use_eng_features and (final_engineered_features is not None and final_engineered_features.shape[1] > 0),
            'n_train_samples': X_train_final.shape[0],
            'n_test_samples': X_test_final.shape[0],
            'window_size': win_size,
            'classes': list(label_encoder.classes_),
            'final_feature_names': final_feature_names, # List of features actually used
            'train_subjects': sorted(list(np.unique(train_subject_ids))),
            'test_subjects': sorted(list(test_subjects))
        }
        output_paths['summary'] = os.path.join(prep_data_dir, 'data_summary.pkl')
        utils.save_pickle(summary, output_paths['summary'])

    except Exception as e:
        logging.error(f"Error saving prepared data artifacts: {e}", exc_info=True)
        raise

    logging.info("Data preparation stage complete.")
    return output_paths


if __name__ == '__main__':
    # Example of how to run this script directly
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        # Assumes config is loaded from ../config.yaml relative to src/
        # Assumes feature_engineering.py outputs are in features/
        # Assumes feature_selector.py output is in results/ (if use_selected_features=True)
        from config_loader import load_config
        script_dir = os.path.dirname(__file__)
        project_root = os.path.abspath(os.path.join(script_dir, '..'))
        config_file_path = os.path.join(project_root, 'config.yaml')
        cfg = load_config(config_file_path)

        utils.set_seed(cfg.get('seed_number', 42))

        # Define input paths based on assumed outputs of previous stages
        feature_dir = os.path.join(project_root, cfg.get('intermediate_feature_dir', 'features'))
        results_dir = os.path.join(project_root, cfg.get('results_dir', 'results'))

        x_raw_path = os.path.join(feature_dir, 'X_windows_raw.npy')
        eng_feat_path = os.path.join(feature_dir, 'engineered_features.npy')
        y_win_path = os.path.join(feature_dir, 'y_windows.npy')
        subj_ids_path = os.path.join(feature_dir, 'subject_ids_windows.npy')
        orig_names_path = os.path.join(feature_dir, 'original_feature_names.pkl')
        eng_names_path = os.path.join(feature_dir, 'engineered_feature_names.pkl')

        sel_feat_path = None
        if cfg.get('use_selected_features', False):
             sel_feat_path = os.path.join(results_dir, cfg.get('feature_selection_input_file', 'selected_features_pyimpetus.pkl'))

        # Check required files exist
        required = [x_raw_path, eng_feat_path, y_win_path, subj_ids_path, orig_names_path, eng_names_path]
        if sel_feat_path: required.append(sel_feat_path)
        if not all(os.path.exists(p) for p in required if p is not None):
            logging.error("One or more input files for data preparation not found.")
            missing = [p for p in required if p is not None and not os.path.exists(p)]
            logging.error(f"Missing files: {missing}")
        else:
            # Execute the main function of this module
            output_artifact_paths = run_data_preparation(
                cfg,
                x_raw_path,
                eng_feat_path,
                y_win_path,
                subj_ids_path,
                orig_names_path,
                eng_names_path,
                sel_feat_path # Pass the path or None
            )
            logging.info("Data preparation complete. Output artifacts:")
            for key, val in output_artifact_paths.items():
                logging.info(f"  {key}: {val}")

    except FileNotFoundError as e:
        logging.error(f"File Not Found Error: {e}")
    except ValueError as e:
        logging.error(f"Value Error: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)