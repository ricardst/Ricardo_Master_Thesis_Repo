# src/data_loader.py

import os
import glob
import pickle
import re
import pandas as pd
import numpy as np
import logging
import gc

# Assuming utils.py is in the same src directory or PYTHONPATH is set
try:
    from . import utils
except ImportError:
    import utils # Fallback if run directly

def run_loading_and_preprocessing(config, input_dir):
    """
    Loads individual subject data, combines, cleans, imputes, and saves.

    Args:
        config (dict): The configuration dictionary.

    Returns:
        str: Path to the saved processed DataFrame file.
        list: List of actual original feature columns found and used.
    """
    input_folder = input_dir
    if not os.path.isdir(input_folder):
        logging.error(f"Input directory for processed subjects not found: {input_folder}")
        raise FileNotFoundError(f"Input directory not found: {input_folder}")

    logging.info(f"Starting data loading from folder: {input_folder}")

    # --- Find and Filter Subject Files ---
    all_pkl_files = glob.glob(os.path.join(input_folder, '*_filtered_corrected*.pkl'))
    if not all_pkl_files:
        logging.error(f"No '*_filtered_corrected*.pkl' files found in {input_folder}")
        raise FileNotFoundError(f"No suitable .pkl files found in {input_folder}")

    # Get the list of subjects to load from config (might be None or empty)
    subjects_to_load = config.get('subjects_to_load', [])

    files_to_load = []
    subjects_found_in_dir = set()
    subject_file_map = {} # Map subject ID to filename for easier filtering

    # First pass: Identify all subjects available in the directory
    for file in all_pkl_files:
        subject_id = utils.extract_subject_id(file)
        if subject_id:
            subjects_found_in_dir.add(subject_id)
            subject_file_map[subject_id] = file

    logging.info(f"Found data for subjects: {sorted(list(subjects_found_in_dir))}")

    # Second pass: Select files based on config
    if subjects_to_load: # If a specific list is provided
        logging.info(f"Config specifies loading ONLY subjects: {subjects_to_load}")
        requested_subjects_set = set(subjects_to_load)
        subjects_actually_loading = []

        # Check which requested subjects are available
        for subject_id in subjects_to_load:
            if subject_id in subject_file_map:
                files_to_load.append(subject_file_map[subject_id])
                subjects_actually_loading.append(subject_id)
            else:
                logging.warning(f"Requested subject '{subject_id}' not found in directory {input_folder}.")

        if not files_to_load:
             logging.error(f"None of the requested subjects {subjects_to_load} were found in {input_folder}.")
             raise FileNotFoundError("No requested subject data files found.")
        logging.info(f"Will load data for {len(subjects_actually_loading)} requested and found subjects.")

    else: # If no specific list, load all found subjects
        logging.info("No specific 'subjects_to_load' in config. Loading all found subjects.")
        files_to_load = all_pkl_files # Use all files found by glob
        subjects_actually_loading = sorted(list(subjects_found_in_dir))

    # --- Load Selected Data ---
    dfs = []
    loaded_subject_ids = set()
    for file in files_to_load:
        subject_id = utils.extract_subject_id(file) # Re-extract for logging clarity
        if subject_id is None: continue # Should not happen if map worked

        logging.info(f"Loading: {os.path.basename(file)} (Subject: {subject_id})")
        try:
            df = utils.load_pickle(file)
            if not isinstance(df, pd.DataFrame) or df.empty:
                 logging.warning(f" Skipping empty or invalid DataFrame in {file}.")
                 continue

            # Add subject ID column (ensure consistent naming via config)
            subj_col_name = config.get('subject_id_column', 'SubjectID')
            df[subj_col_name] = subject_id
            dfs.append(df)
            loaded_subject_ids.add(subject_id)
        except Exception as e:
            logging.error(f"Failed to load or process file {file}: {e}", exc_info=True)
            # Continue trying to load others? Or raise error? Continuing for now.

    if not dfs:
        logging.error("No DataFrames were successfully loaded after filtering/loading.")
        raise ValueError("Failed to load any data.")

    # Concatenate all loaded DataFrames
    logging.info(f"Concatenating data for subjects: {sorted(list(loaded_subject_ids))}...")
    # <<< MODIFIED: Remove ignore_index to preserve original indices (should be DatetimeIndex) >>>
    try:
        data = pd.concat(dfs, ignore_index=False)
        # Sort by index (time) after concatenation
        data.sort_index(inplace=True)
    except Exception as e:
        logging.error(f"Error during concatenation (check if indices overlap or are compatible): {e}", exc_info=True)
        raise ValueError("Failed to concatenate subject DataFrames.")
    # <<< END MODIFIED >>>

    del dfs
    gc.collect()

    logging.info(f"Concatenated data. Total rows: {len(data)}, Index type: {type(data.index)}")
    if isinstance(data.index, pd.DatetimeIndex):
         logging.info(f" Time range: {data.index.min()} to {data.index.max()}")
    else:
         logging.warning(" Index after concatenation is NOT a DatetimeIndex!")


    # --- Remap activity labels using an external mapping file ---
    #mapping_file = config.get('activity_mapping_file', "Activity_Mapping.csv")
    mapping_file = "/scai_data3/scratch/stirnimann_r/Activity_Mapping_v2.csv"
    target_col = config.get('target_column', 'Activity')

    if mapping_file is not None and os.path.exists(mapping_file):
        try:
            mapping_df = pd.read_csv(mapping_file)
            # Create a dictionary for mapping: {Former_Label: New_Label}
            mapping_dict = dict(zip(mapping_df["Former_Label"], mapping_df["New_Label"]))

            if target_col in data.columns:
                logging.info(f"Remapping '{target_col}' labels using {mapping_file}...")
                original_labels = data[target_col].unique()
                data[target_col] = data[target_col].map(mapping_dict).fillna(data[target_col]) # Keep original if no mapping exists
                new_labels = data[target_col].unique()
                logging.info(f"Label remapping complete. Original unique labels: {len(original_labels)}, New unique labels: {len(new_labels)}")
            else:
                logging.warning(f"Target column '{target_col}' not found for remapping. Skipping remapping.")
        except Exception as e:
            logging.error(f"Error during label remapping using {mapping_file}: {e}", exc_info=True)
    else:
        logging.warning(f"Activity mapping file '{mapping_file}' not found. Skipping label remapping.")

    # --- Select feature columns and target column ---
    # Use the list of *original* sensor columns expected at this stage
    X_columns_config = config.get('sensor_columns_original', [])
    y_column = config.get('target_column', 'Activity')
    subj_column = config.get('subject_id_column', 'SubjectID')

    if not X_columns_config:
        logging.error("Configuration key 'sensor_columns_original' is empty or missing.")
        raise ValueError("No original sensor columns specified in configuration.")

    # Verify essential columns exist
    if y_column not in data.columns:
        logging.error(f"Target column '{y_column}' not found in the combined data.")
        # Attempt fallback? For now, raise error.
        raise ValueError(f"Target column '{y_column}' not found in the data")
    if subj_column not in data.columns:
        logging.error(f"Subject ID column '{subj_column}' not found in the combined data.")
        raise ValueError(f"Subject ID column '{subj_column}' not found in the data")

    # Verify subject ID column has no NaNs *before* filtering
    if data[subj_column].isnull().any():
        nan_subj_count = data[subj_column].isnull().sum()
        logging.error(f"Found {nan_subj_count} rows with missing Subject ID in column '{subj_column}'.")
        raise ValueError(f"Missing values found in Subject ID column '{subj_column}'. Please clean the source data.")

    # Verify which sensor columns are actually present
    X_columns_present = [col for col in X_columns_config if col in data.columns]
    missing_cols = [col for col in X_columns_config if col not in data.columns]
    if missing_cols:
        logging.warning(f"Missing original sensor columns in combined data: {missing_cols}")
        logging.warning(f"Present sensor columns in combined data: {X_columns_present}")
    if not X_columns_present:
        logging.error("None of the specified original sensor columns were found in the data.")
        raise ValueError("No valid original sensor columns found in the data")
    logging.info(f"Using {len(X_columns_present)} available original sensor columns as features.")
    X_columns = X_columns_present # Use only the columns that are actually present

    # --- Filter out rows with 'Unknown' or NaN activity label ---
    initial_rows = len(data)
    unknown_label = '' # Define the label to filter

    # Handle potential NaN/None in target column first
    nan_target_mask = data[y_column].isna()
    if nan_target_mask.any():
        logging.warning(f"Found {nan_target_mask.sum()} NaN/None values in target column '{y_column}'. Removing these rows.")
        data = data[~nan_target_mask]

    # Filter by label value
    # Ensure comparison works if labels are not strings
    try:
        data = data[data[y_column] != unknown_label]
    except TypeError:
         logging.warning(f"Target column '{y_column}' might not be string type. Trying conversion for filtering '{unknown_label}'.")
         data = data[data[y_column].astype(str) != unknown_label]

    rows_dropped = initial_rows - len(data)
    if rows_dropped > 0:
        logging.info(f"Dropped {rows_dropped} rows due to NaN target or label '{unknown_label}'. Remaining rows: {len(data)}")
    else:
        logging.info(f"No rows dropped based on target column NaN or label '{unknown_label}'.")

    if data.empty:
        logging.error("Data is empty after filtering labels. Check input data and labels.")
        raise ValueError("No data remaining after label filtering.")

    # --- Impute Missing Values in Feature Columns ---
    logging.info(f"Checking for missing values in {len(X_columns)} feature columns before imputation...")
    initial_nans = data[X_columns].isnull().sum()
    total_initial_nans = initial_nans.sum()

    if total_initial_nans > 0:
        logging.warning(f"Found {total_initial_nans} missing values across feature columns.")
        logging.debug(f"NaN counts per column:\n{initial_nans[initial_nans > 0]}")
        logging.info("Applying forward fill (ffill) followed by backward fill (bfill) imputation...")
        # Apply ffill first
        data[X_columns] = data[X_columns].fillna(method='ffill')
        # Apply bfill next
        data[X_columns] = data[X_columns].fillna(method='bfill')
        # Check if any NaNs remain (should only happen if a column is entirely NaN at start/end of a subject's data)
        remaining_nans = data[X_columns].isnull().sum().sum()
        if remaining_nans > 0:
            logging.error(f"Found {remaining_nans} missing values *after* ffill/bfill imputation.")
            logging.error("This likely means NaNs exist at the very start or end of the data for some columns.")
            logging.warning("Filling these remaining NaNs with 0.")
            data[X_columns] = data[X_columns].fillna(0) # Fill remaining with 0
            # Alternatively, raise error:
            # raise ValueError("NaN values remain in feature columns after ffill/bfill imputation.")
        else:
            logging.info("Imputation complete. No remaining NaNs detected in feature columns.")
    else:
        logging.info("No missing values found in feature columns. Skipping imputation.")

    # --- Prepare Final Output ---
    # Select only the necessary columns for the next stage
    final_columns = [subj_column, y_column] + X_columns
    processed_data = data[final_columns].copy()

    # Define output path
    output_dir = config.get('intermediate_feature_dir', 'features')
    os.makedirs(output_dir, exist_ok=True)
    output_filename = "combined_cleaned_data.pkl"
    output_path = os.path.join(output_dir, output_filename)

    # Define output path for the list of columns used
    output_filename_cols = "original_feature_names.pkl"
    output_path_cols = os.path.join(output_dir, output_filename_cols)

    # Save the processed data
    logging.info(f"Saving combined, cleaned, imputed data to {output_path}...")
    utils.save_pickle(processed_data, output_path)
    logging.info(f"Saving original feature names list to {output_path_cols}...")
    utils.save_pickle(X_columns, output_path_cols)

    return output_path, X_columns # Return path and the list of actual feature columns used

if __name__ == '__main__':
    # Example of how to run this script directly
    # Requires config_loader.py and utils.py
    # Assumes config.yaml is in the parent directory

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        from config_loader import load_config
    except ImportError:
        print("Error: Could not import load_config. Make sure config_loader.py is accessible.")
        exit()

    try:
        script_dir = os.path.dirname(__file__)
        project_root = os.path.abspath(os.path.join(script_dir, '..')) # Assumes src is one level down
        config_file_path = os.path.join(project_root, 'config.yaml')
        cfg = load_config(config_file_path)

        # Set seed using utils
        utils.set_seed(cfg.get('seed_number', 42))

        # Execute the main function of this module
        output_file_path, feature_cols = run_loading_and_preprocessing(cfg, input_dir=cfg.get('cleaned_data_input_dir'))
        logging.info(f"Data loading and preprocessing complete. Output saved to: {output_file_path}")
        logging.info(f"Feature columns used: {feature_cols}")

    except FileNotFoundError as e:
        logging.error(f"File Not Found Error: {e}")
    except ValueError as e:
        logging.error(f"Value Error: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)