# src/feature_engineering.py

import logging
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy import fft as sp_fft
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
import gc # For garbage collection between batches

# Assuming utils.py is in the same src directory or PYTHONPATH is set
try:
    from . import utils
except ImportError:
     # Fallback if run directly
     import utils # Ensure utils can be found if run directly and is in PYTHONPATH

# Attempt CuPy Import and Check Availability (revised for clarity and correctness)
CUPY_AVAILABLE = False
cp = None
cupy_fft_module = None # Renamed to avoid conflict if cp.fft is preferred
cupy_stats_module = None # For stats_module.entropy

try:
    import cupy as cp
    if cp.is_available():
        CUPY_AVAILABLE = True
        logging.info("CuPy available for feature engineering.")
        # Assign modules to be used later if CUPY_AVAILABLE
        import cupy.fft as cp_fft # Use cp.fft directly or this alias
        cupy_fft_module = cp_fft
        import cupyx.scipy.stats as cp_scipy_stats # For entropy, changed from cupy.scipy.stats
        cupy_stats_module = cp_scipy_stats
    else:
        CUPY_AVAILABLE = False # Ensure it's False if not available
        logging.warning("CuPy installed, but no compatible GPU found. Using NumPy/SciPy.")
except ImportError:
    CUPY_AVAILABLE = False
    logging.info("CuPy not found. Using NumPy/SciPy for feature engineering.")


def create_windows(data, feature_cols, target_col, subject_col, config):
    """
    Creates sliding windows from the time series data.
    (Copied from original script, ensuring float32 output)
    """
    window_size = config.get('window_size', 100)
    window_step = config.get('window_step', 50)

    logging.info(f"Creating windows: size={window_size}, step={window_step}...")

    if not isinstance(data.index, pd.DatetimeIndex):
        logging.error("Input data for create_windows must have a DatetimeIndex.")
        raise TypeError("Input data index is not DatetimeIndex.")
    
    # Extract numpy arrays for features, target, and subject IDs for efficiency
    try:
        X = data[feature_cols].values
        y = data[target_col].values
        subj_ids = data[subject_col].values
    except KeyError as e:
        logging.error(f"Missing expected column in input data for windowing: {e}")
        raise
    n_samples = len(X)

    if n_samples < window_size:
        logging.warning(f"Data length ({n_samples}) < window size ({window_size}). Cannot create windows.")
        return np.array([]).reshape(0, window_size, len(feature_cols)), np.array([]), np.array([]), np.array([], dtype='datetime64[ns]'), np.array([], dtype='datetime64[ns]')

    n_windows = (n_samples - window_size) // window_step + 1
    logging.info(f"Attempting to create {n_windows} windows from {n_samples} samples.")

    X_windows = np.zeros((n_windows, window_size, len(feature_cols)), dtype=np.float32)
    y_windows = np.zeros(n_windows, dtype=object) # Use object to allow for np.nan or strings
    subject_ids_windows = np.zeros(n_windows, dtype=object)
    start_times = np.zeros(n_windows, dtype='datetime64[ns]')
    end_times = np.zeros(n_windows, dtype='datetime64[ns]')

    actual_windows_created = 0
    for i in range(n_windows):
        start_idx = i * window_step
        end_idx = start_idx + window_size

        if end_idx > n_samples: 
            break # Should not happen with n_windows calculation, but as safeguard

        X_windows[actual_windows_created] = X[start_idx:end_idx].astype(np.float32)
        subject_ids_windows[actual_windows_created] = subj_ids[start_idx]
        start_times[actual_windows_created] = data.index[start_idx].to_numpy()
        end_times[actual_windows_created] = data.index[end_idx - 1].to_numpy() # End timestamp is the last sample IN the window


        window_y_series = pd.Series(y[start_idx:end_idx])
        modes = window_y_series.dropna().mode()

        if not modes.empty:
            y_windows[actual_windows_created] = modes.iloc[0] # Take the first mode if multiple
        else:
            y_windows[actual_windows_created] = np.nan # Assign NaN if no mode (all NaN or empty after dropna)
            

        actual_windows_created += 1

    # Trim if any were skipped
    if actual_windows_created < n_windows:
        logging.info(f"Trimming window arrays from {n_windows} to actual created size: {actual_windows_created}")
        X_windows = X_windows[:actual_windows_created]
        y_windows = y_windows[:actual_windows_created]
        subject_ids_windows = subject_ids_windows[:actual_windows_created]
        start_times = start_times[:actual_windows_created]
        end_times = end_times[:actual_windows_created]


    if actual_windows_created == 0:
        logging.warning("No windows were successfully created.")
        # Return empty arrays with correct dimensions for consistency
        return np.array([]).reshape(0, window_size, len(feature_cols)), np.array([]), np.array([]), np.array([], dtype='datetime64[ns]'), np.array([], dtype='datetime64[ns]')


    logging.info(f"Successfully created {actual_windows_created} windows.")
    return X_windows, y_windows, subject_ids_windows, start_times, end_times


def plot_windows_to_pdf(X_windows, y_windows, subject_ids_windows, window_start_times, feature_names, config, base_output_dir="window_plots"):
    """
    Plots every 100th window of sensor data to a PDF file, one sensor per plot, one plot per page.
    A separate PDF is created for each subject.
    Includes window start time in the plot title.
    """
    logging.info(f"Starting to plot windows. Output directory: {base_output_dir}")
    os.makedirs(base_output_dir, exist_ok=True)

    unique_subjects = np.unique(subject_ids_windows)
    sampling_rate = config.get('downsample_freq', 25) # Get sampling rate from config for time axis

    for subject_id in unique_subjects:
        subject_mask = (subject_ids_windows == subject_id)
        X_subject = X_windows[subject_mask]
        y_subject = y_windows[subject_mask]
        start_times_subject = window_start_times[subject_mask] # Filter start times for the subject

        if X_subject.shape[0] == 0:
            logging.info(f"No windows to plot for subject {subject_id}.")
            continue

        pdf_path = os.path.join(base_output_dir, f"{subject_id}_sensor_windows.pdf")
        with PdfPages(pdf_path) as pdf:
            logging.info(f"Creating PDF for subject {subject_id} at {pdf_path}")
            # Iterate through every 100th window for the current subject
            for window_idx in range(0, X_subject.shape[0], 100):
                window_data = X_subject[window_idx] # Shape: (window_size, n_features)
                window_label = y_subject[window_idx]
                current_window_start_time = start_times_subject[window_idx]
                
                # Format window start time
                try:
                    # Convert numpy.datetime64 to pandas Timestamp for easy formatting
                    window_time_str = pd.Timestamp(current_window_start_time).strftime('%Y-%m-%d %H:%M:%S')
                except Exception as e:
                    logging.warning(f"Could not format timestamp {current_window_start_time}: {e}")
                    window_time_str = "Unknown Time"

                # Time axis for the plot
                time_axis = np.arange(window_data.shape[0]) / sampling_rate

                # One plot per sensor (feature)
                for feature_idx in range(window_data.shape[1]):
                    sensor_data = window_data[:, feature_idx]
                    sensor_name = feature_names[feature_idx]

                    plt.figure(figsize=(11.69, 8.27)) # A4 landscape
                    plt.plot(time_axis, sensor_data)
                    plt.title(f"Subject: {subject_id} - Window: {window_idx // 100 + 1} (Raw index: {window_idx}) - Time: {window_time_str}\\nSensor: {sensor_name} - Label: {window_label}")
                    plt.xlabel(f"Time (seconds) within window (SR: {sampling_rate} Hz)")
                    plt.ylabel("Sensor Value")
                    plt.grid(True)
                    pdf.savefig() # Saves the current figure into a new page in the PDF
                    plt.close() # Close the figure to free memory
            logging.info(f"Finished PDF for subject {subject_id}.")
    logging.info("Finished plotting all windows.")


def engineer_features_for_windows(X_windows, feature_names, sampling_rate):
    """
    Calculates time, frequency, and pressure features for each window, using CuPy if available.
    (Copied from previous step, includes CuPy, float32, skew/kurtosis handling)
    """
    xp = cp if CUPY_AVAILABLE and cp is not None else np
    # Use the imported modules/functions directly
    xp_fft = cupy_fft_module if CUPY_AVAILABLE and cupy_fft_module is not None else sp_fft
    current_stats_module = cupy_stats_module if CUPY_AVAILABLE and cupy_stats_module is not None else sp_stats

    using_gpu = CUPY_AVAILABLE and cp is not None # Check cp too
    dtype_to_use = xp.float32

    if using_gpu:
        logging.debug("Using CuPy (GPU) for feature calculations.")
        X_windows_xp = xp.asarray(X_windows, dtype=dtype_to_use)
    else:
        logging.debug("Using NumPy/SciPy (CPU) for feature calculations.")
        X_windows_xp = xp.asarray(X_windows, dtype=dtype_to_use) # This will be np.asarray if not using_gpu

    n_windows, window_size, n_original_features = X_windows_xp.shape
    engineered_features_list = []
    engineered_feature_names = []
    logging.debug(f"Starting feature engineering for {n_windows} windows in this batch...")

    # --- Time Domain ---
    logging.debug("Calculating time-domain features...")
    mean_vals = xp.mean(X_windows_xp, axis=1)
    std_vals = xp.std(X_windows_xp, axis=1)
    min_vals = xp.min(X_windows_xp, axis=1)
    max_vals = xp.max(X_windows_xp, axis=1)
    range_vals = max_vals - min_vals
    rms_vals = xp.sqrt(xp.mean(xp.square(X_windows_xp), axis=1))
    median_vals = xp.median(X_windows_xp, axis=1)

    logging.debug("Calculating IQR, Skew, Kurtosis, ZCR...")
    percentile_75 = xp.percentile(X_windows_xp, 75, axis=1)
    percentile_25 = xp.percentile(X_windows_xp, 25, axis=1)
    iqr_vals = percentile_75 - percentile_25

    if using_gpu:
        logging.debug(" Transferring data to CPU for SciPy skew/kurtosis calculation.")
        X_windows_np = X_windows_xp.get() # cupy.ndarray.get()
        skew_vals_np = sp_stats.skew(X_windows_np, axis=1)
        kurt_vals_np = sp_stats.kurtosis(X_windows_np, axis=1)
        skew_vals = cp.asarray(skew_vals_np, dtype=dtype_to_use)
        kurt_vals = cp.asarray(kurt_vals_np, dtype=dtype_to_use)
        del X_windows_np # Clean up CPU copy
    else:
        skew_vals = sp_stats.skew(X_windows_xp, axis=1).astype(dtype_to_use)
        kurt_vals = sp_stats.kurtosis(X_windows_xp, axis=1).astype(dtype_to_use)

    zcr_vals = xp.mean(((X_windows_xp[:, 1:, :] * X_windows_xp[:, :-1, :]) < 0), axis=1)

    engineered_features_list.extend([
        mean_vals, std_vals, min_vals, max_vals, range_vals, rms_vals, median_vals,
        iqr_vals, skew_vals, kurt_vals, zcr_vals
    ])
    time_feature_suffixes = ['_mean','_std','_min','_max','_range','_rms','_median','_iqr','_skew','_kurt','_zcr']
    for suffix in time_feature_suffixes:
        engineered_feature_names.extend([f"{name}{suffix}" for name in feature_names])

    # --- Frequency Domain ---
    logging.debug("Calculating frequency-domain features...")
    # stats_module is now current_stats_module
    fft_coeffs = xp_fft.fft(X_windows_xp, axis=1) # Use xp_fft.fft
    fft_freqs = xp_fft.fftfreq(window_size, 1.0 / sampling_rate) # Use xp_fft.fftfreq
    fft_freqs_xp = xp.asarray(fft_freqs) # Ensure it's on the correct device (xp)
    positive_freq_mask = fft_freqs_xp > 0
    fft_freqs_pos = fft_freqs_xp[positive_freq_mask]

    # Handle case where no positive frequencies exist (e.g., window_size=1)
    if fft_freqs_pos.size == 0:
        logging.warning("No positive frequencies found for FFT. Skipping frequency features for this batch.")
        # Add placeholders if needed, or skip appending freq features
        dominant_freq_vals = xp.zeros((n_windows, n_original_features), dtype=dtype_to_use) # Changed to dtype_to_use
        spectral_energy_vals = xp.zeros((n_windows, n_original_features), dtype=dtype_to_use)
        spectral_entropy_vals = xp.zeros((n_windows, n_original_features), dtype=dtype_to_use)
    else:
        # Ensure fft_coeffs is (n_windows, n_freqs, n_features)
        # positive_freq_mask is 1D, apply it to the frequency axis (axis=1)
        fft_coeffs_pos = fft_coeffs[:, positive_freq_mask, :]
        fft_magnitudes = xp.abs(fft_coeffs_pos)
        fft_psd = xp.square(fft_magnitudes) / window_size # PSD definition can vary, this is one common way
        dominant_freq_index = xp.argmax(fft_magnitudes, axis=1)
        # dominant_freq_vals needs to be (n_windows, n_original_features)
        # fft_freqs_pos is (n_positive_freqs,). dominant_freq_index is (n_windows, n_original_features)
        # We need to gather elements carefully.
        # Assuming dominant_freq_vals should be one value per feature per window:
        dominant_freq_vals = fft_freqs_pos[dominant_freq_index]

        spectral_energy_vals = xp.sum(fft_psd, axis=1)
        logging.debug("Calculating spectral entropy...")
        psd_sum = xp.sum(fft_psd, axis=1, keepdims=True)
        epsilon = xp.array(1e-7, dtype=dtype_to_use) # Epsilon for float32 stability, on correct device
        normalized_psd = fft_psd / (psd_sum + epsilon)
        # Ensure normalized_psd is compatible with stats_module.entropy
        # Scipy's entropy expects probabilities (sum to 1).
        # If using cupy.scipy.stats.entropy, it should be similar.
        spectral_entropy_vals = current_stats_module.entropy(normalized_psd.astype(dtype_to_use), base=2, axis=1)


    engineered_features_list.extend([
        dominant_freq_vals.astype(dtype_to_use), spectral_energy_vals, spectral_entropy_vals.astype(dtype_to_use)
    ])
    freq_feature_suffixes = ['_dom_freq', '_spec_energy', '_spec_entropy']
    for suffix in freq_feature_suffixes:
        engineered_feature_names.extend([f"{name}{suffix}" for name in feature_names])

    # --- Pressure Specific ---
    logging.debug("Calculating pressure-specific features...")
    # ... (Pressure feature calculation using xp remains the same) ...
    bottom_cols = [i for i, name in enumerate(feature_names) if name.startswith('bottom_value_')]
    back_cols = [i for i, name in enumerate(feature_names) if name.startswith('back_value_')]
    mean_pressure_bottom = None
    mean_pressure_back = None
    if bottom_cols:
        X_bottom = X_windows_xp[:, :, bottom_cols]
        mean_pressure_bottom = xp.mean(xp.mean(X_bottom, axis=2), axis=1, keepdims=True) # Ensure dtype
        mean_max_pressure_bottom = xp.mean(xp.max(X_bottom, axis=2), axis=1, keepdims=True)
        mean_var_pressure_bottom = xp.mean(xp.var(X_bottom, axis=2), axis=1, keepdims=True)
        engineered_features_list.extend([
            mean_pressure_bottom.astype(dtype_to_use),
            mean_max_pressure_bottom.astype(dtype_to_use),
            mean_var_pressure_bottom.astype(dtype_to_use)
        ])
        engineered_feature_names.extend(['pressure_bottom_mean', 'pressure_bottom_mean_max', 'pressure_bottom_mean_var'])
    if back_cols:
        X_back = X_windows_xp[:, :, back_cols]
        mean_pressure_back = xp.mean(xp.mean(X_back, axis=2), axis=1, keepdims=True) # Ensure dtype
        mean_max_pressure_back = xp.mean(xp.max(X_back, axis=2), axis=1, keepdims=True)
        mean_var_pressure_back = xp.mean(xp.var(X_back, axis=2), axis=1, keepdims=True)
        engineered_features_list.extend([
            mean_pressure_back.astype(dtype_to_use),
            mean_max_pressure_back.astype(dtype_to_use),
            mean_var_pressure_back.astype(dtype_to_use)
        ])
        engineered_feature_names.extend(['pressure_back_mean', 'pressure_back_mean_max', 'pressure_back_mean_var'])

    if mean_pressure_bottom is not None and mean_pressure_back is not None:
         pressure_diff_mean = mean_pressure_back - mean_pressure_bottom
         engineered_features_list.append(pressure_diff_mean.astype(dtype_to_use))
         engineered_feature_names.append('pressure_diff_back_minus_bottom_mean')


    # --- Combine ---
    if not engineered_features_list: # Check if list is empty
         logging.warning("No engineered features were calculated for this batch.")
         # Return an empty array with expected second dimension if no features, or handle upstream
         return np.array([]).reshape(n_windows, 0), []


    all_engineered_features_xp = xp.concatenate(engineered_features_list, axis=1)
    logging.debug(f"Finished feature engineering for batch. Added {all_engineered_features_xp.shape[1]} features.")

    if using_gpu:
        logging.debug("Transferring batch results from GPU back to CPU...")
        all_engineered_features_np = all_engineered_features_xp.get()
        del X_windows_xp, all_engineered_features_xp # Explicitly delete CuPy arrays on GPU
        if hasattr(cp, 'get_default_memory_pool'): # Optional cleanup
            cp.get_default_memory_pool().free_all_blocks()
        return all_engineered_features_np, engineered_feature_names # Return np array
    else:
        return all_engineered_features_xp, engineered_feature_names # This is already a NumPy array if not using_gpu


def run_feature_engineering(input_dataframe_path, config):
    """
    Loads cleaned data, creates windows, engineers features, and saves outputs.
    """
    logging.info(f"--- Starting Feature Engineering Stage ---")
    logging.info(f"Loading cleaned data from: {input_dataframe_path}")
    data = utils.load_pickle(input_dataframe_path)

    if not isinstance(data, pd.DataFrame) or data.empty:
        logging.error("Loaded data is not a valid or non-empty DataFrame.")
        raise ValueError("Invalid input data for feature engineering.")

    # Get column names from config or data
    # It's safer to get feature columns from the actual data loaded
    feature_cols = config.get('sensor_columns_original', [])
    target_col = config.get('target_column', 'Activity')
    subject_col = config.get('subject_id_column', 'SubjectID')

    # Verify columns exist in the loaded data
    actual_feature_cols = [col for col in feature_cols if col in data.columns]
    missing_cols = [col for col in feature_cols if col not in data.columns]
    if missing_cols:
        logging.warning(f"Columns specified in config missing from loaded data: {missing_cols}")
    if not actual_feature_cols:
         raise ValueError("No specified feature columns found in the loaded data.")
    if target_col not in data.columns:
         raise ValueError(f"Target column '{target_col}' not found.")
    if subject_col not in data.columns:
         raise ValueError(f"Subject column '{subject_col}' not found.")

    logging.info(f"Using {len(actual_feature_cols)} features for windowing.")

    # --- Create Windows (including timestamps) ---
    X_windows_raw, y_windows, subject_ids_windows, window_start_times, window_end_times = create_windows(
        data, actual_feature_cols, target_col, subject_col, config
    )

    # --- Plotting Windows (New) ---
    plot_config = config.get('plotting_config', {})
    if plot_config.get('enable_window_plotting', False):
        logging.info("Window plotting enabled. Generating plots...")
        plot_output_dir = plot_config.get('window_plots_output_dir', 'window_plots') # Default to 'window_plots'
        # Ensure the base directory for plots exists (e.g., results/)
        base_results_dir = config.get('results_dir', '.') # Get general results_dir or use current
        full_plot_output_dir = os.path.join(base_results_dir, plot_output_dir)
        
        plot_windows_to_pdf(
            X_windows_raw, 
            y_windows, 
            subject_ids_windows, 
            window_start_times, # Pass the window start times
            actual_feature_cols, # Use the actual feature columns used for windowing
            config, # Pass the main config for sampling rate etc.
            base_output_dir=full_plot_output_dir
        )
    else:
        logging.info("Window plotting is disabled in the configuration.")

    # --- Handle NaN Window Labels ---
    nan_label_mask = pd.isna(y_windows)
    num_nan_labels = np.sum(nan_label_mask)
    if num_nan_labels > 0:
        logging.warning(f"Found {num_nan_labels} windows with NaN labels (from mode calculation). Removing them.")
        X_windows_raw = X_windows_raw[~nan_label_mask]
        y_windows = y_windows[~nan_label_mask]
        subject_ids_windows = subject_ids_windows[~nan_label_mask]
        window_start_times = window_start_times[~nan_label_mask] 
        window_end_times = window_end_times[~nan_label_mask]     


    if X_windows_raw.shape[0] == 0:
        logging.error("No valid windows remaining after filtering NaN labels.")
        raise ValueError("No windows left after label filtering.")
    logging.info(f"Kept {X_windows_raw.shape[0]} windows after label filtering.")

    calculate_features = config.get('calculate_engineered_features', True) # Default True
    engineered_features = None
    final_engineered_feature_names = []

    if calculate_features:
        logging.info("Calculating engineered features...")
        # --- Feature Engineering in Batches ---
        FEAT_ENG_BATCH_SIZE = config.get('feat_eng_batch_size', 10000)
        n_total_windows = X_windows_raw.shape[0]
        all_engineered_features_list = []
        sampling_rate = config.get('downsample_freq', 25)

        logging.info(f" Starting calculation in batches of size {FEAT_ENG_BATCH_SIZE}...")
        for i in range(0, n_total_windows, FEAT_ENG_BATCH_SIZE):
            # ... (batching loop logic: get X_batch_raw, log progress) ...
            start_idx = i
            end_idx = min(i + FEAT_ENG_BATCH_SIZE, n_total_windows)
            X_batch_raw = X_windows_raw[start_idx:end_idx]
            logging.info(f"  Processing batch {i // FEAT_ENG_BATCH_SIZE + 1}...")

            engineered_features_batch, engineered_feature_names_batch = engineer_features_for_windows(
                X_batch_raw,
                actual_feature_cols,
                sampling_rate
            )
            if i == 0: final_engineered_feature_names = engineered_feature_names_batch
            if engineered_features_batch.shape[1] > 0: all_engineered_features_list.append(engineered_features_batch)
            gc.collect()
            if CUPY_AVAILABLE:
                try: cp.get_default_memory_pool().free_all_blocks()
                except Exception: pass

        # --- Combine and Clean Engineered Features ---
        if all_engineered_features_list:
            engineered_features = np.concatenate(all_engineered_features_list, axis=0)
            logging.info(f" Concatenated engineered features. Shape: {engineered_features.shape}")
            # Clean potential NaNs/Infs
            nan_count = np.isnan(engineered_features).sum()
            inf_count = np.isinf(engineered_features).sum()
            if nan_count > 0 or inf_count > 0:
                logging.warning(f" Found {nan_count} NaNs and {inf_count} Infs in engineered features. Replacing with 0.0.")
                engineered_features = np.nan_to_num(engineered_features, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            logging.warning("Engineered feature calculation resulted in no features.")
            # Fallback to creating an empty array if calculation ran but yielded nothing
            engineered_features = np.empty((n_total_windows, 0), dtype=np.float32)

    else:
        # Skip calculation, create empty placeholders
        logging.info("Skipping engineered feature calculation based on config.")
        n_total_windows = X_windows_raw.shape[0]
        engineered_features = np.empty((n_total_windows, 0), dtype=np.float32) # Empty array with 0 features
        final_engineered_feature_names = [] # Empty list

    # --- Save Outputs ---
    output_dir = config.get('intermediate_feature_dir', 'features')
    os.makedirs(output_dir, exist_ok=True)

    output_paths = {}
    try:
        # Always save raw windows, labels, subjects, and original names
        output_paths['X_windows_raw'] = os.path.join(output_dir, 'X_windows_raw.npy')
        utils.save_numpy(X_windows_raw, output_paths['X_windows_raw'])
        output_paths['y_windows'] = os.path.join(output_dir, 'y_windows.npy')
        utils.save_numpy(y_windows, output_paths['y_windows'])
        output_paths['subject_ids_windows'] = os.path.join(output_dir, 'subject_ids_windows.npy')
        utils.save_numpy(subject_ids_windows, output_paths['subject_ids_windows'])
        output_paths['original_feature_names'] = os.path.join(output_dir, 'original_feature_names.pkl')
        utils.save_pickle(actual_feature_cols, output_paths['original_feature_names']) # Save actual cols used

        # Save engineered features (even if empty) and their names (even if empty list)
        output_paths['engineered_features'] = os.path.join(output_dir, 'engineered_features.npy')
        utils.save_numpy(engineered_features, output_paths['engineered_features'])
        output_paths['engineered_feature_names'] = os.path.join(output_dir, 'engineered_feature_names.pkl')
        utils.save_pickle(final_engineered_feature_names, output_paths['engineered_feature_names'])

        # Save timestamps
        output_paths['window_start_times'] = os.path.join(output_dir, 'window_start_times.npy')
        utils.save_numpy(window_start_times, output_paths['window_start_times'])
        output_paths['window_end_times'] = os.path.join(output_dir, 'window_end_times.npy')
        utils.save_numpy(window_end_times, output_paths['window_end_times'])


        logging.info(f"All feature engineering outputs saved to directory: {output_dir}")
    except Exception as e:
        logging.error(f"Error saving feature engineering outputs: {e}", exc_info=True)
        raise

    return output_paths


if __name__ == '__main__':
    # Example of how to run this script directly
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        # Assumes config is loaded from ../config.yaml relative to src/
        # Assumes data_loader.py has run and produced combined_cleaned_data.pkl in features/
        from config_loader import load_config
        script_dir = os.path.dirname(__file__)
        project_root = os.path.abspath(os.path.join(script_dir, '..'))
        config_file_path = os.path.join(project_root, 'config.yaml')
        cfg = load_config(config_file_path)

        utils.set_seed(cfg.get('seed_number', 42))

        # Define input path based on assumed output of data_loader stage
        input_df_path = os.path.join(project_root, cfg.get('intermediate_feature_dir', 'features'), 'combined_cleaned_data.pkl')

        if not os.path.exists(input_df_path):
            logging.error(f"Input DataFrame not found for feature engineering: {input_df_path}")
            logging.error("Please run the data_loader stage first.")
        else:
            # Execute the main function of this module
            output_file_paths = run_feature_engineering(input_df_path, cfg)
            logging.info(f"Feature engineering complete. Output files saved:")
            for key, val in output_file_paths.items():
                logging.info(f"  {key}: {val}")

    except FileNotFoundError as e:
        logging.error(f"File Not Found Error: {e}")
    except ValueError as e:
        logging.error(f"Value Error: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)