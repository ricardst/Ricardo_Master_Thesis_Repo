# src/raw_data_processor.py

import os
import pandas as pd
import numpy as np
from datetime import timedelta
from scipy.signal import butter, sosfiltfilt # Use SOS for stability
import yaml
import time
import sys
import gc
import logging
import re

# Assuming utils.py is in the same src directory or PYTHONPATH is set
try:
    from . import utils
except ImportError:
    # If run directly, try importing assuming src is in PYTHONPATH or relative
    try:
        import utils
    except ImportError:
        print("ERROR: Failed to import 'utils' module. Ensure it's in the 'src' directory or PYTHONPATH.")
        sys.exit(1)

# Import for plotting
try:
    from matplotlib.backends.backend_pdf import PdfPages
except ImportError:
    logging.warning("Matplotlib PdfPages could not be imported. Plotting to PDF will not be available.")
    PdfPages = None # Define it as None so checks can be made

# --- Core Helper Functions ---
# These functions perform specific processing steps and should be kept.
# They can be imported by the analysis notebook later.

def correct_timestamp_drift(sensor_time, t0_numeric, t1_numeric, total_drift):
    """Adjusts sensor time (seconds) for linear drift."""

    total_interval = t1_numeric - t0_numeric

    if total_interval == 0: return sensor_time

    elapsed = sensor_time - t0_numeric
    drift_offset = total_drift * (elapsed / total_interval)
    
    return sensor_time + drift_offset

def process_file_numeric_time(file_path, data_list):
    """Helper to load CSV and append to list if valid."""
    if os.path.getsize(file_path) > 0:
        try:
            df = pd.read_csv(file_path, compression='gzip')
            if not df.empty and 'time' in df.columns:
                data_list.append(df)
        except pd.errors.EmptyDataError:
             logging.debug(f" EmptyDataError loading {file_path}. Skipping.") # Use debug
        except Exception as e:
            logging.error(f" Error loading {file_path}: {e}")

def data_loader_no_dir(subject_dir, modality, settings):
    """Loads data for modalities where files are directly in the modality folder."""
    modality_dir = os.path.join(subject_dir, modality)
    data = []
    if os.path.exists(modality_dir):
        files = [f for f in os.listdir(modality_dir) if f.endswith(tuple(settings.get('file_format', ['.csv.gz'])))]
        for file_name in files:
            file_path = os.path.join(modality_dir, file_name)
            process_file_numeric_time(file_path, data)
    if data:
        df_modality = pd.concat(data, ignore_index=True).sort_values(by=['time']).reset_index(drop=True)
        columns_to_keep = ['time'] + settings.get('data_columns', [])
        cols_in_df = [col for col in columns_to_keep if col in df_modality.columns]
        if len(cols_in_df) <= 1: return pd.DataFrame()
        missing_cols = [col for col in columns_to_keep if col not in df_modality.columns]
        if missing_cols: logging.warning(f" Columns missing in {modality}: {missing_cols}.")
        return df_modality[cols_in_df]
    return pd.DataFrame()

def data_loader_with_dir(subject_dir, modality, settings):
    """Loads data for modalities where files might be in subdirectories."""
    modality_dir = os.path.join(subject_dir, modality)
    data = []
    if not os.path.exists(modality_dir): return pd.DataFrame()
    for root, _, files in os.walk(modality_dir):
        for file_name in files:
            if file_name.endswith(tuple(settings.get('file_format', ['.csv.gz']))):
                file_path = os.path.join(root, file_name)
                process_file_numeric_time(file_path, data)
    if data:
        df_modality = pd.concat(data, ignore_index=True).sort_values(by=['time']).reset_index(drop=True)
        columns_to_keep = ['time'] + settings.get('data_columns', [])
        cols_in_df = [col for col in columns_to_keep if col in df_modality.columns]
        if len(cols_in_df) <= 1: return pd.DataFrame()
        missing_cols = [col for col in columns_to_keep if col not in df_modality.columns]
        if missing_cols: logging.warning(f" Columns missing in {modality} (subdirs): {missing_cols}.")
        return df_modality[cols_in_df]
    return pd.DataFrame()

def data_loader_no_dir_ms(subject_dir, modality, settings):
    """Uses data_loader_no_dir. Time unit conversion handled later."""
    return data_loader_no_dir(subject_dir, modality, settings)

def select_data_loader(sensor_name):
    """Selects the appropriate data loader based on sensor name convention."""
    if sensor_name == 'vivalnk_vv330_acceleration': return data_loader_with_dir
    # Add other specific cases if needed
    return data_loader_no_dir # Default

def process_modality_duplicates(data_df, sample_rate):
    """Handles duplicate timestamps by adding small increments using integer positions."""
    # ... (Implementation remains the same as previous correct version) ...
    if data_df.empty or not isinstance(data_df.index, pd.DatetimeIndex): return data_df
    if not data_df.index.is_monotonic_increasing: data_df = data_df.sort_index()
    duplicates_mask = data_df.index.duplicated(keep=False)
    if not duplicates_mask.any(): return data_df
    logging.debug(f"Adjusting {np.sum(data_df.index.duplicated())} duplicate timestamp entries...")
    new_index_values = data_df.index.to_numpy(copy=True)
    time_delta_ns = int(1e9 / sample_rate) if sample_rate > 0 else 0
    time_delta = np.timedelta64(time_delta_ns, 'ns')
    dup_int_indices = np.where(duplicates_mask)[0]
    count_dict = {}
    for idx in dup_int_indices:
        current_ts = new_index_values[idx]
        if idx > 0 and current_ts == new_index_values[idx-1]:
            count = count_dict.get(current_ts, 0) + 1
            new_index_values[idx] = new_index_values[idx] + count * time_delta
            count_dict[current_ts] = count
        else:
            count_dict[current_ts] = 0
    try:
        data_df.index = pd.DatetimeIndex(new_index_values)
        data_df = data_df[~data_df.index.duplicated(keep='first')]
    except Exception as e:
        logging.error(f"Error applying adjusted index: {e}. Removing duplicates directly.", exc_info=True)
        return data_df[~data_df.index.duplicated(keep='first')]
    return data_df


def handle_missing_data_interpolation(data_df, max_interp_gap_s, target_freq):
    """Interpolates short gaps in data with DatetimeIndex."""
    # ... (Implementation remains the same) ...
    if data_df.empty or not isinstance(data_df.index, pd.DatetimeIndex): return data_df
    if not data_df.index.is_monotonic_increasing: data_df = data_df.sort_index()
    time_diff = data_df.index.to_series().diff().dt.total_seconds()
    long_gaps = time_diff > max_interp_gap_s
    if long_gaps.any(): logging.debug(f"Found {long_gaps.sum()} gaps > {max_interp_gap_s}s (won't be time-interpolated).")
    temp_df = data_df.copy(); temp_df.loc[long_gaps, :] = np.nan
    try:
        limit_val = int(max_interp_gap_s * target_freq) + 1 if target_freq > 0 else None
        interpolated_df = temp_df.interpolate(method='time', limit=limit_val, limit_direction='both')
        remaining_nan_count = interpolated_df.isnull().sum().sum()
        if remaining_nan_count > 0: logging.debug(f"Filling {remaining_nan_count} remaining NaNs with 0 after interpolation.")
        return interpolated_df # Let NaNs pass through
    except Exception as e:
        logging.error(f"Error during interpolation: {e}. Filling all NaNs with 0.", exc_info=True)
        return data_df.fillna(0)

def modify_modality_names(data, sensor_name):
    """Renames columns based on sensor name using a predefined map."""
    # ... (Implementation remains the same - ensure your map is correct) ...
    rename_map = { # Make sure keys match sensor_name from config
        'corsano_wrist_acc': {'accX': 'wrist_acc_x', 'accY': 'wrist_acc_y', 'accZ': 'wrist_acc_z', '_new_prefix': 'corsano_wrist'},
        'cosinuss_ear_acc_x_acc_y_acc_z': {'acc_x': 'ear_acc_x', 'acc_y': 'ear_acc_y', 'acc_z': 'ear_acc_z', '_new_prefix': 'cosinuss_ear'},
        'mbient_imu_wc_accelerometer': {'x-axis (g)': 'imu_acc_x', 'y-axis (g)': 'imu_acc_y', 'z-axis (g)': 'imu_acc_z', '_new_prefix': 'mbient_acc'},
        'mbient_imu_wc_gyroscope': {'x-axis (deg/s)': 'gyro_x', 'y-axis (deg/s)': 'gyro_y', 'z-axis (deg/s)': 'gyro_z', '_new_prefix': 'mbient_gyro'},
        'vivalnk_vv330_acceleration': {'x': 'vivalnk_acc_x', 'y': 'vivalnk_acc_y', 'z': 'vivalnk_acc_z', '_new_prefix': 'vivalnk_acc'},
        'sensomative_bottom_logger': {**{f'value_{i}': f'bottom_value_{i}' for i in range(1, 12)}, '_new_prefix': 'sensomative_bottom'},
        'sensomative_back_logger': {**{f'value_{i}': f'back_value_{i}' for i in range(1, 12)}, '_new_prefix': 'sensomative_back'},
        'corsano_bioz_acc': {'accX': 'bioz_acc_x', 'accY': 'bioz_acc_y', 'accZ': 'bioz_acc_z', '_new_prefix': 'corsano_bioz'},
    }
    mapping = rename_map.get(sensor_name.lower())
    if mapping:
        new_prefix = mapping.pop('_new_prefix', sensor_name)
        current_cols_lower = {c.lower(): c for c in data.columns}
        rename_dict_case_insensitive = {current_cols_lower[k.lower()]: v for k, v in mapping.items() if k.lower() in current_cols_lower}
        if rename_dict_case_insensitive: # Check if any columns actually match
             data = data.rename(columns=rename_dict_case_insensitive, errors='ignore')
        return new_prefix, data
    else:
        logging.debug(f"No specific renaming rule found for '{sensor_name}'. Returning original.")
        return sensor_name, data


def butter_lowpass_sos(cutoff, fs, order=5):
    """Creates Butterworth lowpass filter coefficients using SOS format."""
    # ... (Implementation remains the same) ...
    nyq = 0.5 * fs
    if cutoff >= nyq: cutoff = nyq * 0.99
    if cutoff <= 0: return None
    normal_cutoff = cutoff / nyq
    try: return butter(order, normal_cutoff, btype='low', analog=False, output='sos')
    except ValueError as e: logging.error(f"Error creating Butterworth SOS filter: {e}"); return None

def apply_filter_combined(df_to_modify, sos, columns_to_filter):
    """
    Applies filtfilt using SOS to specified columns of a DataFrame IN-PLACE.
    """
    if sos is None:
        logging.debug("SOS filter None, skipping filtering (no changes to DataFrame).")
        return df_to_modify
    if df_to_modify.empty or not columns_to_filter:
        logging.debug("DataFrame empty or no columns to filter (no changes to DataFrame).")
        return df_to_modify

    valid_cols_to_filter = [col for col in columns_to_filter if col in df_to_modify.columns]

    if not valid_cols_to_filter:
        logging.warning("No valid columns to filter found in DataFrame after checking existence.")
        return df_to_modify

    logging.info(f"Applying filter column by column IN-PLACE to {len(valid_cols_to_filter)} columns...")
    
    original_col_for_error = "Unknown" # For logging in case of error
    try:
        for col_idx, col in enumerate(valid_cols_to_filter):
            original_col_for_error = col # Update for current iteration
            logging.debug(f"Filtering column {col_idx + 1}/{len(valid_cols_to_filter)}: {col}")
            
            # Extract, fill NaNs, and convert to float64 for filtering. This creates a temporary Series.
            # This is a copy of ONE column's data.
            temp_col_series = df_to_modify[col].fillna(0).astype(np.float64)
            temp_col_values = temp_col_series.values

            min_len_for_filter = 7 
            if len(temp_col_values) < min_len_for_filter:
                logging.warning(f"Skipping filter for column {col} due to insufficient data length ({len(temp_col_values)} < {min_len_for_filter}). Assigning 0-filled data back.")
                df_to_modify[col] = temp_col_series # Assign back the 0-filled (but unfiltered) series
                continue 
            
            if temp_col_values.ndim == 1 and len(temp_col_values) > 0:
                filtered_col_values = sosfiltfilt(sos, temp_col_values)
                df_to_modify[col] = filtered_col_values # Assign filtered data back to the original DataFrame
            else:
                logging.warning(f"Column {col} is empty or not 1D after processing, skipping filter. Assigning 0-filled data back.")
                df_to_modify[col] = temp_col_series # Assign back the 0-filled (but unfiltered) series
            
            if (col_idx + 1) % 10 == 0: 
                logging.debug(f"Filtered {col_idx + 1}/{len(valid_cols_to_filter)} columns.")
        
        gc.collect()

    except Exception as e:
        # Log with the specific column that caused the error
        logging.error(f"Error applying filter for column {original_col_for_error}: {e}. This column in the DataFrame may not be correctly filtered or may retain pre-error state.", exc_info=True)
        # The DataFrame df_to_modify is returned as is, potentially partially modified.

    return df_to_modify


# --- Main Subject Processing Function ---
def process_subject_data(subject_id, config, correction_params, global_labels_df, project_root_dir):
    logging.info(f"--- Processing Subject: {subject_id} ---")

    raw_data_base_dir = os.path.join(project_root_dir, config.get('raw_data_input_dir'))
    subject_dir = os.path.join(raw_data_base_dir, subject_id)
    raw_data_parsing_config = config.get('raw_data_parsing_config', {})
    subject_correction_params = correction_params.get(subject_id, {})
    downsample_freq = config.get('downsample_freq', 20)
    filter_params = config.get('filter_parameters', {})
    highcut = filter_params.get('highcut_kinematic', 9.9)
    filter_order = filter_params.get('filter_order', 4)

    enable_plots_global = config.get('enable_preprocessing_plots', False)
    plots_output_dir_config = config.get('preprocessing_plots_output_dir', 'results/preprocessing_visualizations')
    if not os.path.isabs(plots_output_dir_config):
        plots_output_dir = os.path.join(project_root_dir, plots_output_dir_config)
    else:
        plots_output_dir = plots_output_dir_config
        
    plot_chunk_minutes = config.get('plot_time_chunk_minutes', 5)
    plot_label_colors = config.get('plot_label_colors', None)

    if enable_plots_global and PdfPages is not None:
        try:
            os.makedirs(plots_output_dir, exist_ok=True)
            logging.info(f"[{subject_id}] Plotting enabled. Output directory: {plots_output_dir}")
        except Exception as e:
            logging.error(f"[{subject_id}] Failed to create plots output directory: {e}. Plotting will be disabled.", exc_info=True)
            enable_plots_global = False
    elif enable_plots_global and PdfPages is None:
        logging.warning("Plotting was enabled, but PdfPages (matplotlib) is not available. Plotting will be skipped.")
        enable_plots_global = False

    subject_labels = global_labels_df[global_labels_df['Video_File'].str.contains(subject_id, na=False)].copy()
    if not subject_labels.empty:
        subject_labels['Real_Start_Time'] = pd.to_datetime(subject_labels['Real_Start_Time'], errors='coerce')
        subject_labels['Real_End_Time'] = pd.to_datetime(subject_labels['Real_End_Time'], errors='coerce')
        subject_labels.dropna(subset=['Real_Start_Time', 'Real_End_Time'], inplace=True)
    
    if not subject_labels.empty:
        label_time_shift_str = subject_correction_params.get('Label_Time_Shift', '0h 0min 0s')
        shift_match = re.match(r'(?:(-?\d+)h)?\s*(?:(-?\d+)min)?\s*(?:(-?\d+)s)?', label_time_shift_str)
        if shift_match:
            shift_hours = int(shift_match.group(1) or 0); shift_minutes = int(shift_match.group(2) or 0); shift_seconds = int(shift_match.group(3) or 0)
            total_shift_seconds = (shift_hours * 3600) + (shift_minutes * 60) + shift_seconds
            if total_shift_seconds != 0:
                time_delta_shift = pd.Timedelta(seconds=total_shift_seconds)
                subject_labels['Real_Start_Time'] += time_delta_shift
                subject_labels['Real_End_Time'] += time_delta_shift
    
    min_label_time = subject_labels['Real_Start_Time'].min() if not subject_labels.empty else pd.NaT
    max_label_time = subject_labels['Real_End_Time'].max() if not subject_labels.empty else pd.NaT
    filter_min_label_time_adj = min_label_time - pd.Timedelta(hours=3) if pd.notna(min_label_time) else pd.NaT
    filter_max_label_time_adj = max_label_time + pd.Timedelta(hours=3) if pd.notna(max_label_time) else pd.NaT

    processed_sensors_dict = {}
    min_corrected_time, max_corrected_time = pd.NaT, pd.NaT

    for sensor_name_orig, sensor_settings in raw_data_parsing_config.items():
        logging.debug(f"[{subject_id}] Loading/Processing sensor: {sensor_name_orig}")
        loader = select_data_loader(sensor_name_orig)
        sensor_data_raw = loader(subject_dir, sensor_name_orig, sensor_settings)
        if sensor_data_raw.empty or 'time' not in sensor_data_raw.columns: continue

        try:
            sensor_corr_params = subject_correction_params.get(sensor_name_orig, {'unit': 's'})
            time_unit = sensor_corr_params.get('unit', 's')
            time_col_num = sensor_data_raw['time'].astype(float)
            time_col_num_shifted = time_col_num / 1000.0 if time_unit=='ms' else time_col_num
            time_col_num_shifted += sensor_corr_params.get('shift', 0)
            drift_params = sensor_corr_params.get('drift'); time_col_final_num = time_col_num_shifted
            if drift_params and all(k in drift_params for k in ['t0', 't1', 'drift_secs']):
                t0_ts, t1_ts = pd.Timestamp(drift_params['t0']), pd.Timestamp(drift_params['t1'])
                if pd.isna(t0_ts) or pd.isna(t1_ts):
                    logging.warning(f"[{subject_id}] Invalid drift t0/t1 for {sensor_name_orig}. Skipping drift correction.")
                else:
                    t0, t1, drift = t0_ts.timestamp(), t1_ts.timestamp(), drift_params['drift_secs']
                    time_col_final_num = time_col_num_shifted.apply(correct_timestamp_drift, args=(t0, t1, drift))
            
            corrected_timestamps = pd.to_datetime(time_col_final_num, unit='s', errors='coerce')
            sensor_data_indexed = sensor_data_raw.drop(columns=['time'])
            sensor_data_indexed['time'] = corrected_timestamps
            
            if 'time' not in sensor_data_indexed.columns: continue
            if not pd.api.types.is_datetime64_any_dtype(sensor_data_indexed['time']):
                sensor_data_indexed['time'] = pd.to_datetime(sensor_data_indexed['time'], errors='coerce')
            sensor_data_indexed.dropna(subset=['time'], inplace=True)
            if sensor_data_indexed.empty: continue

            if pd.notna(filter_min_label_time_adj): sensor_data_indexed = sensor_data_indexed[sensor_data_indexed['time'] >= filter_min_label_time_adj]
            if pd.notna(filter_max_label_time_adj): sensor_data_indexed = sensor_data_indexed[sensor_data_indexed['time'] <= filter_max_label_time_adj]
            if sensor_data_indexed.empty: continue
            
            sensor_data_indexed.set_index('time', inplace=True)
        except Exception as e:
            logging.error(f"[{subject_id}] Error during time correction for {sensor_name_orig}: {e}", exc_info=True)
            continue

        sample_rate = sensor_settings.get('sample_rate', downsample_freq)
        processed_data_intermediate = process_modality_duplicates(sensor_data_indexed, sample_rate)
        processed_data_intermediate = handle_missing_data_interpolation(processed_data_intermediate, max_interp_gap_s=2, target_freq=downsample_freq)
        if processed_data_intermediate.empty: continue

        new_name, modified_data_synced = modify_modality_names(processed_data_intermediate, sensor_name_orig) # Renamed to modified_data_synced
        if modified_data_synced.empty: continue
        
        safe_new_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', new_name)

        if enable_plots_global and PdfPages is not None and not modified_data_synced.empty:
            pdf_path_synced = os.path.join(plots_output_dir, f"{subject_id}_{safe_new_name}_synced.pdf")
            try:
                with PdfPages(pdf_path_synced) as pdf_writer_synced:
                    plot_title = f"Subject: {subject_id} - Synced - Sensor: {new_name}"
                    logging.debug(f"Plotting to {pdf_path_synced}")
                    utils.plot_timeseries_to_pdf(
                        dataframe=modified_data_synced,
                        pdf_pages_object=pdf_writer_synced,
                        title_base=plot_title,
                        subject_labels_df=subject_labels,
                        columns_to_plot=modified_data_synced.columns.tolist(),
                        chunk_duration_minutes=plot_chunk_minutes,
                        plot_label_colors=plot_label_colors
                    )
            except Exception as e_plot:
                logging.error(f"[{subject_id}] Error plotting synced data for {new_name} to {pdf_path_synced}: {e_plot}", exc_info=True)

        processed_sensors_dict[new_name] = modified_data_synced # Store the synced data
        sensor_min = modified_data_synced.index.min(); sensor_max = modified_data_synced.index.max()
        if pd.notna(sensor_min): min_corrected_time = min(min_corrected_time, sensor_min) if pd.notna(min_corrected_time) else sensor_min
        if pd.notna(sensor_max): max_corrected_time = max(max_corrected_time, sensor_max) if pd.notna(max_corrected_time) else sensor_max

    if not processed_sensors_dict: return None, None
    
    # Create a map of sensor identifier (new_name) to its original columns after syncing
    # These are the columns that originated from this sensor group.
    sensor_to_original_columns_map = {
        name: df.columns.tolist() for name, df in processed_sensors_dict.items()
    }

    abs_min_time = min(min_corrected_time, min_label_time) if pd.notna(min_corrected_time) and pd.notna(min_label_time) else (min_corrected_time if pd.notna(min_corrected_time) else min_label_time)
    abs_max_time = max(max_corrected_time, max_label_time) if pd.notna(max_corrected_time) and pd.notna(max_label_time) else (max_corrected_time if pd.notna(max_corrected_time) else max_label_time)
    if pd.isna(abs_min_time) or pd.isna(abs_max_time): return None, None
    target_freq_interval = pd.Timedelta(seconds=1.0 / downsample_freq)
    final_start_time = pd.Timestamp(abs_min_time).floor(target_freq_interval)
    final_end_time = pd.Timestamp(abs_max_time).ceil(target_freq_interval)
    if pd.isna(final_start_time) or pd.isna(final_end_time) or (final_end_time - final_start_time).total_seconds() <= 0: return None, None
    duration_seconds = (final_end_time - final_start_time).total_seconds()
    max_duration_days = config.get('max_processing_duration_days', 7)
    if duration_seconds > max_duration_days * 24 * 60 * 60: return None, None
    target_aligned_index = pd.date_range(start=final_start_time, end=final_end_time, freq=target_freq_interval)
    if target_aligned_index.empty: return None, None

    combined_data_resampled = pd.DataFrame(index=target_aligned_index) 
    try:
        resampled_sensors_list = []
        for name, df_sensor in processed_sensors_dict.items(): # Use original processed_sensors_dict for resampling
            if df_sensor.empty: continue
            resampled_df = df_sensor.resample(target_freq_interval).mean().reindex(target_aligned_index)
            if not resampled_df.empty:
                resampled_sensors_list.append(resampled_df)
        
        # Del processed_sensors_dict after use if not needed for column mapping, but it IS needed.
        # No, keep processed_sensors_dict for sensor_to_original_columns_map if created above, or create map earlier.
        # sensor_to_original_columns_map is created *from* processed_sensors_dict, so this is fine.
        # We can del processed_sensors_dict itself if its DataFrames are large and map is enough.
        # For now, let's assume memory is fine.

        if not resampled_sensors_list: return None, None
        combined_data_resampled = pd.concat(resampled_sensors_list, axis=1)
        del resampled_sensors_list; gc.collect() 
    except Exception as e:
        logging.error(f"[{subject_id}] Error during resample/reindex/concat: {e}", exc_info=True)
        return None, None
    if combined_data_resampled.empty: return None, None

    # --- PLOTTING STAGE 2: After Resampling (Per Original Sensor Group) ---
    if enable_plots_global and PdfPages is not None and not combined_data_resampled.empty:
        for sensor_name_key, original_cols in sensor_to_original_columns_map.items():
            # Select only columns that are actually present in combined_data_resampled AND belong to this sensor
            cols_for_this_sensor_in_resampled = [col for col in original_cols if col in combined_data_resampled.columns]
            if not cols_for_this_sensor_in_resampled:
                logging.debug(f"[{subject_id}] No columns for sensor {sensor_name_key} found in resampled data. Skipping resampled plot for this sensor.")
                continue
            
            resampled_sensor_subset_df = combined_data_resampled[cols_for_this_sensor_in_resampled].copy()
            if resampled_sensor_subset_df.empty: continue

            safe_sensor_name_key = re.sub(r'[^a-zA-Z0-9_\-]', '_', sensor_name_key)
            pdf_path_resampled_sensor = os.path.join(plots_output_dir, f"{subject_id}_{safe_sensor_name_key}_resampled.pdf")
            try:
                with PdfPages(pdf_path_resampled_sensor) as pdf_writer_resampled_sensor:
                    plot_title = f"Subject: {subject_id} - Resampled - Sensor: {sensor_name_key} (Pre-Imputation)"
                    logging.debug(f"Plotting to {pdf_path_resampled_sensor}")
                    utils.plot_timeseries_to_pdf(
                        dataframe=resampled_sensor_subset_df,
                        pdf_pages_object=pdf_writer_resampled_sensor,
                        title_base=plot_title,
                        subject_labels_df=subject_labels,
                        columns_to_plot=cols_for_this_sensor_in_resampled,
                        chunk_duration_minutes=plot_chunk_minutes,
                        plot_label_colors=plot_label_colors
                    )
            except Exception as e_plot:
                logging.error(f"[{subject_id}] Error plotting resampled data for sensor {sensor_name_key} to {pdf_path_resampled_sensor}: {e_plot}", exc_info=True)
    # --- END PLOTTING STAGE 2 ---

    combined_data_imputed = combined_data_resampled.copy()
    combined_data_imputed.ffill(limit=int(downsample_freq*2), inplace=True)
    combined_data_imputed.bfill(limit=int(downsample_freq*2), inplace=True)
    combined_data_imputed.fillna(0, inplace=True)
    gc.collect()
    
    if combined_data_imputed.columns.duplicated().any():
        combined_data_imputed = combined_data_imputed.loc[:,~combined_data_imputed.columns.duplicated()]
    
    feature_columns_list = sorted([col for col in combined_data_imputed.columns if col not in [config.get('target_column', 'Activity'), config.get('subject_id_column', 'SubjectID')]])

    sos = butter_lowpass_sos(highcut, downsample_freq, filter_order)
    combined_data_filtered = combined_data_imputed.copy()
    if feature_columns_list and not combined_data_filtered.empty and sos is not None:
        combined_data_filtered = apply_filter_combined(combined_data_filtered, sos, feature_columns_list)
    gc.collect()

    # --- PLOTTING STAGE 3: After Filtering (Per Original Sensor Group) ---
    if enable_plots_global and PdfPages is not None and not combined_data_filtered.empty:
        for sensor_name_key, original_cols in sensor_to_original_columns_map.items():
            # Select columns for this sensor that are among the actual feature_columns_list AND present in filtered data
            cols_for_this_sensor_to_plot_filtered = [
                col for col in original_cols 
                if col in combined_data_filtered.columns and col in feature_columns_list
            ]
            if not cols_for_this_sensor_to_plot_filtered:
                logging.debug(f"[{subject_id}] No columns for sensor {sensor_name_key} found in filtered data / feature list. Skipping filtered plot for this sensor.")
                continue
            
            filtered_sensor_subset_df = combined_data_filtered[cols_for_this_sensor_to_plot_filtered].copy()
            if filtered_sensor_subset_df.empty: continue
            
            safe_sensor_name_key = re.sub(r'[^a-zA-Z0-9_\-]', '_', sensor_name_key)
            pdf_path_filtered_sensor = os.path.join(plots_output_dir, f"{subject_id}_{safe_sensor_name_key}_filtered.pdf")
            try:
                with PdfPages(pdf_path_filtered_sensor) as pdf_writer_filtered_sensor:
                    plot_title = f"Subject: {subject_id} - Filtered - Sensor: {sensor_name_key}"
                    logging.debug(f"Plotting to {pdf_path_filtered_sensor}")
                    utils.plot_timeseries_to_pdf(
                        dataframe=filtered_sensor_subset_df,
                        pdf_pages_object=pdf_writer_filtered_sensor,
                        title_base=plot_title,
                        subject_labels_df=subject_labels,
                        columns_to_plot=cols_for_this_sensor_to_plot_filtered,
                        chunk_duration_minutes=plot_chunk_minutes,
                        plot_label_colors=plot_label_colors
                    )
            except Exception as e_plot:
                logging.error(f"[{subject_id}] Error plotting filtered data for sensor {sensor_name_key} to {pdf_path_filtered_sensor}: {e_plot}", exc_info=True)
    # --- END PLOTTING STAGE 3 ---

    final_data = combined_data_filtered 
    target_col = config.get('target_column', 'Activity')
    final_data[target_col] = 'Unknown'

    if not subject_labels.empty:
        for _, row in subject_labels.iterrows():
            start_ts = row['Real_Start_Time'] 
            end_ts = row['Real_End_Time']     
            activity = row['Label']
            if pd.isna(start_ts) or pd.isna(end_ts): continue
            if start_ts.tzinfo is not None: start_ts = start_ts.tz_localize(None)
            if end_ts.tzinfo is not None: end_ts = end_ts.tz_localize(None)
            try:
                final_data.loc[start_ts:end_ts, target_col] = activity
            except Exception as e:
                logging.error(f"[{subject_id}] Error applying label '{activity}' for range {start_ts}-{end_ts}: {e}")
    gc.collect()

    subject_col_cfg = config.get('subject_id_column', 'SubjectID')
    final_data[subject_col_cfg] = subject_id
    final_columns_list_output = feature_columns_list + [target_col, subject_col_cfg]
    
    current_cols_in_final_data = final_data.columns.tolist()
    for col_to_add in final_columns_list_output:
        if col_to_add not in current_cols_in_final_data:
            final_data[col_to_add] = 0 
            
    try: 
        final_data = final_data[final_columns_list_output]
    except KeyError as e:
        logging.error(f"[{subject_id}] KeyError reindexing final_data. Missing: {e}. Available: {final_data.columns.tolist()}")
        return None, None

    logging.info(f"[{subject_id}] Final processed data shape: {final_data.shape}. Columns: {final_data.columns.tolist()}")
    return final_data, feature_columns_list


# --- Main Orchestration Function ---
def run_raw_processing(config, project_root_dir):
    """Main function for the raw data processing stage."""
    logging.info("--- Starting Raw Data Processing Stage ---")

    # --- Setup (Load Labels, Sync Params, Subjects, Dirs) ---
    try:
        global_labels_path = os.path.join(project_root_dir, config.get('global_labels_file'))
        global_labels = pd.read_csv(global_labels_path, parse_dates=['Real_Start_Time', 'Real_End_Time'])
        
        sync_params_path = os.path.join(project_root_dir, config.get('sync_parameters_file'))
        with open(sync_params_path, 'r') as f: correction_params = yaml.safe_load(f)
    except Exception as e: 
        logging.error(f"Error loading global labels/sync params: {e}", exc_info=True)
        raise

    subjects = config.get('subjects_to_process', [])
    if not subjects: 
        logging.error("No subjects to process specified in config.")
        raise ValueError("No subjects to process in config.")
        
    output_dir_config = config.get('processed_data_output_dir', 'processed_subjects')
    if not os.path.isabs(output_dir_config):
        output_dir = os.path.join(project_root_dir, output_dir_config)
    else:
        output_dir = output_dir_config
    os.makedirs(output_dir, exist_ok=True)

    all_processed_feature_columns = set()

    # --- Process Each Subject ---
    for subject in subjects:
        subject_data, feature_cols_list = process_subject_data(
            subject, config, correction_params, global_labels, project_root_dir # Pass project_root_dir
        )
        if subject_data is None or subject_data.empty: 
            logging.warning(f"No data returned for subject {subject}. Skipping save.")
            continue

        if feature_cols_list:
             all_processed_feature_columns.update(feature_cols_list)

        # --- Save Output PKL ---
        output_file = os.path.join(output_dir, f"{subject}_filtered_corrected.pkl")
        try:
            logging.info(f"Saving final processed data for {subject} to {output_file}")
            utils.save_pickle(subject_data, output_file)
        except Exception as e: 
            logging.error(f"Error saving pickle for {subject}: {e}", exc_info=True)
        del subject_data; gc.collect()

    # --- Save Metadata ---
    metadata = {
        'subjects_processed': subjects,
        'processing_timestamp': pd.Timestamp.now().isoformat(),
        'config_parameters_used': {
            'downsample_freq': config.get('downsample_freq'),
            'filter_parameters': config.get('filter_parameters'),
        },
        'final_feature_column_names': sorted(list(all_processed_feature_columns))
    }
    metadata_file = os.path.join(output_dir, 'raw_processing_metadata.pkl') # Save in the same output dir
    utils.save_pickle(metadata, metadata_file)
    logging.info(f"Saved raw processing metadata to {metadata_file}")

    logging.info("--- Raw Data Processing Stage Complete ---")
    return output_dir


# --- if __name__ block ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    script_dir_main = os.path.dirname(os.path.abspath(__file__))
    project_root_main = os.path.abspath(os.path.join(script_dir_main, '..'))

    try:
        try:
            from config_loader import load_config
        except ImportError:
             print("ERROR: Ensure config_loader.py is in the src directory or PYTHONPATH")
             sys.exit(1)

        config_file_path = os.path.join(project_root_main, 'config.yaml')
        cfg = load_config(config_file_path)

        utils.set_seed(cfg.get('seed_number', 42))

        processed_dir = run_raw_processing(cfg, project_root_main)
        logging.info(f"Raw data processing complete. Output saved to directory: {processed_dir}")

    except FileNotFoundError as e: logging.error(f"File Not Found Error: {e}")
    except ValueError as e: logging.error(f"Value Error: {e}")
    except Exception as e: logging.error(f"An unexpected error occurred: {e}", exc_info=True)