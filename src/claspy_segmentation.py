import os
import pandas as pd
import numpy as np
import yaml
import logging
import random
from typing import Union, List, Tuple, Dict, Any, Optional # Added for type hinting compatibility

from claspy.segmentation import BinaryClaSPSegmentation # Assuming claspy is installed
from matplotlib.backends.backend_pdf import PdfPages # For potential future use, not primary for .png
import matplotlib.pyplot as plt # For direct plot saving if needed

# --- Configuration Area (User Modifiable) ---

# Columns for Claspy Segmentation: List of prefixes or full names.
# Claspy will use data from columns in the input DataFrame that start with these prefixes.
# Example: ['wrist_acc_', 'ear_acc_'] will use all 'wrist_acc_x,y,z' and 'ear_acc_x,y,z' columns.
# Example: ['imu_acc_x', 'imu_acc_y', 'imu_acc_z'] for specific columns.
SEGMENTATION_COLUMN_PREFIXES = ['wrist_acc_', 'x_axis_', 'y_axis_', 'z_axis_', 'vivalnk_acc_', 'bottom_value_'] # General example
# SEGMENTATION_COLUMN_PREFIXES = ['wrist_acc_x', 'wrist_acc_y', 'wrist_acc_z'] # More specific example

# Sensor Modality for Plotting: Prefix for the sensor modality to display in plots.
# The plot will show all channels associated with this prefix (e.g., 'corsano_wrist_' -> 'corsano_wrist_acc_x', 'corsano_wrist_acc_y', 'corsano_wrist_acc_z').
PLOTTING_MODALITY_PREFIX = 'wrist_'
# PLOTTING_MODALITY_PREFIX = 'vivalnk_acc_'

# Claspy Parameters for BinaryClaSPSegmentation
# Refer to Claspy documentation for options: https://claspy.readthedocs.io/en/latest/user_guide/03_profiles.html
CLASPY_PARAMS = {
    'n_jobs' : 16,
    #'n_changepoints': None,      # Let Claspy determine the number of changepoints
    #'period_length': 'auto',     # Or an integer, e.g., 25*60*10 (if 25Hz, for 10 min period)
    # 'window_size': 100,        # Example: if you want to set a specific window size
    # 'verbose': 0
}

# Plotting Configuration
NUM_RANDOM_FRAMES = 5
FRAME_DURATION_HOURS = 0.5
BASE_PLOT_OUTPUT_DIR_NAME = 'claspy_segmentation_plots' # Directory in project root

# Data Configuration
ACTIVITY_COLUMN_NAME = 'Activity' # As defined in raw_data_processor.py
UNKNOWN_ACTIVITY_LABEL = 'Unknown' # Label for segments without a specific activity
DEFAULT_SAMPLING_RATE = 25 # Hz, fallback if not in config (but should be from config.downsample_freq)

# --- End Configuration Area ---

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper Functions ---

def get_project_root() -> str:
    """Determines the project root directory assuming the script is in src/ or similar."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def load_main_config(project_root_dir: str) -> Dict[str, Any]:
    """Loads the main config.yaml file."""
    config_path = os.path.join(project_root_dir, 'config.yaml')
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Successfully loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"ERROR: Main configuration file not found at {config_path}")
        raise
    except Exception as e:
        logger.error(f"ERROR: Could not load or parse main configuration file {config_path}: {e}")
        raise

def load_sync_parameters(project_root_dir: str) -> Dict[str, Any]:
    """Loads the Sync_Parameters.yaml file."""
    sync_params_path = os.path.join(project_root_dir, 'Sync_Parameters.yaml')
    try:
        with open(sync_params_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Successfully loaded sync parameters from {sync_params_path}")
        return config
    except FileNotFoundError:
        logger.warning(f"WARNING: Sync parameters file not found at {sync_params_path}. Label time shifts will not be available.")
        return {}
    except Exception as e:
        logger.error(f"ERROR: Could not load or parse sync parameters file {sync_params_path}: {e}")
        return {}

def load_subject_data(subject_id: str, pkl_file_path: str) -> Optional[pd.DataFrame]: # Changed type hint
    """Loads the _filtered_corrected.pkl file for a subject."""
    if not os.path.exists(pkl_file_path):
        logger.warning(f"Pickle file not found for subject {subject_id} at {pkl_file_path}")
        return None
    try:
        df = pd.read_pickle(pkl_file_path)
        logger.info(f"Successfully loaded data for subject {subject_id} from {pkl_file_path}. Shape: {df.shape}")
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning(f"DataFrame index for {subject_id} is not DatetimeIndex. Trying to convert.")
            df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        logger.error(f"Error loading pickle file for subject {subject_id} at {pkl_file_path}: {e}")
        return None

def select_data_columns(df: pd.DataFrame, column_prefixes_or_names: List[str]) -> pd.DataFrame:
    """Selects columns from df based on a list of prefixes or full names."""
    selected_cols = []
    df_cols_lower = [col.lower() for col in df.columns]
    original_df_cols = df.columns.tolist()

    for pattern in column_prefixes_or_names:
        pattern_lower = pattern.lower()
        matched_full_names = [original_df_cols[i] for i, col_low in enumerate(df_cols_lower) if col_low == pattern_lower]
        if matched_full_names:
            selected_cols.extend(matched_full_names)
        else:
            matched_prefixes = [original_df_cols[i] for i, col_low in enumerate(df_cols_lower) if col_low.startswith(pattern_lower)]
            selected_cols.extend(matched_prefixes)
    
    selected_cols = list(dict.fromkeys(selected_cols))
    
    if not selected_cols:
        logger.warning(f"No columns found matching patterns: {column_prefixes_or_names}")
        return pd.DataFrame()
        
    logger.info(f"Selected columns for processing: {selected_cols}")
    return df[selected_cols]

def get_ground_truth_changepoints(df: pd.DataFrame, activity_col_name: str, unknown_label: str) -> List[int]:
    """Converts the 'Activity' column to a list of ground truth changepoint indices."""
    if activity_col_name not in df.columns:
        logger.warning(f"Activity column '{activity_col_name}' not found. Cannot determine ground truth changepoints.")
        return []
    
    activity_series = df[activity_col_name]
    change_indices = activity_series.ne(activity_series.shift()).to_numpy()
    gt_cps = np.where(change_indices)[0].tolist()

    if 0 in gt_cps:
        gt_cps.remove(0)
        
    logger.info(f"Derived {len(gt_cps)} ground truth changepoints from '{activity_col_name}'.")
    return gt_cps

def select_random_time_frames(df_length: int, num_frames: int, frame_duration_samples: int,
                              activity_series: Optional[pd.Series] = None, # Changed type hint
                              unknown_label: str = UNKNOWN_ACTIVITY_LABEL) -> List[Tuple[int, int]]:
    """Selects start and end indices for random time frames."""
    selected_frames = []
    possible_starts = []

    if df_length < frame_duration_samples:
        logger.warning(f"Data length ({df_length}) is less than frame duration ({frame_duration_samples}). Cannot select frames.")
        if df_length > 0 : return [(0, df_length-1)] 
        return []

    if activity_series is not None and not activity_series.empty:
        has_known_activity = (activity_series != unknown_label) & (~activity_series.isna())
        for i in range(df_length - frame_duration_samples + 1):
            if has_known_activity.iloc[i : i + frame_duration_samples].any():
                possible_starts.append(i)
    
    if not possible_starts: 
        logger.info("No frames with known activities found, or activity_series not provided. Selecting from all possible frames.")
        possible_starts = list(range(df_length - frame_duration_samples + 1))

    if not possible_starts:
        logger.warning("No possible start indices for frames. Data might be too short.")
        return []

    num_to_select = min(num_frames, len(possible_starts))
    if num_to_select < num_frames:
        logger.warning(f"Could only find {num_to_select} possible frames, requested {num_frames}.")
    
    random_starts = random.sample(possible_starts, num_to_select)
    
    for start_idx in random_starts:
        end_idx = start_idx + frame_duration_samples -1 
        selected_frames.append((start_idx, end_idx))
        
    logger.info(f"Selected {len(selected_frames)} random frames for plotting.")
    return selected_frames

def save_changepoints_to_csv(changepoints: Union[List[int], np.ndarray], subject_id: str, output_dir: str):
    """Saves detected changepoints to a CSV file."""
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{subject_id}_detected_changepoints.csv")
    try:
        pd.DataFrame(changepoints, columns=['changepoint_index']).to_csv(filename, index=False)
        logger.info(f"Detected changepoints for {subject_id} saved to {filename}")
    except Exception as e:
        logger.error(f"Error saving changepoints for {subject_id} to {filename}: {e}")

def plot_segmented_frame(
    claspy_model: BinaryClaSPSegmentation,
    full_ts_data_for_plotting: pd.DataFrame, 
    all_detected_cps: List[int],
    all_gt_cps: List[int],
    frame_start_idx: int,
    frame_end_idx: int,
    subject_id: str,
    plot_modality_name: str, 
    frame_num: int,
    output_dir: str,
    activity_labels_full: pd.Series, 
    sampling_rate: float
):
    """Plots a single 1-hour frame using clasp.plot()."""
    os.makedirs(output_dir, exist_ok=True)
    
    frame_data = full_ts_data_for_plotting.iloc[frame_start_idx : frame_end_idx + 1]
    if frame_data.empty:
        logger.warning(f"Frame {frame_num} for {subject_id} is empty. Skipping plot.")
        return

    frame_detected_cps = [cp - frame_start_idx for cp in all_detected_cps if frame_start_idx < cp < frame_end_idx]
    frame_gt_cps = [cp - frame_start_idx for cp in all_gt_cps if frame_start_idx < cp < frame_end_idx]

    activities_in_frame = activity_labels_full.iloc[frame_start_idx : frame_end_idx + 1].unique()
    activities_in_frame_str = ", ".join(sorted([act for act in activities_in_frame if act != UNKNOWN_ACTIVITY_LABEL and pd.notna(act)]))
    if not activities_in_frame_str:
        activities_in_frame_str = "Unknown/No Specific Activity"

    plot_title = (f"Segmentation: {subject_id} - Frame {frame_num + 1}\n"
                  f"Modality: {plot_modality_name} (Channels: {', '.join(frame_data.columns.tolist())})\n"
                  f"Activities: {activities_in_frame_str}")
    
    output_filename = os.path.join(output_dir, f"{subject_id}_{plot_modality_name.replace('_','')}_frame{frame_num + 1}.png")

    try:
        ts_to_plot_np = frame_data.to_numpy()
        
        if ts_to_plot_np.ndim == 1:
            ts_to_plot_np = ts_to_plot_np.reshape(-1, 1)
        
        logger.debug(f"Plotting frame {frame_num+1} for {subject_id}. Data shape for plot: {ts_to_plot_np.shape}")
        logger.debug(f"Frame CPs (detected): {frame_detected_cps}, Frame CPs (GT): {frame_gt_cps}")

        fig, ax = claspy_model.plot(
            gt_cps=frame_gt_cps,            
            heading=plot_title,
            ts_name=plot_modality_name, 
            font_size=10,
            fig_size=(15, max(5, 2 * ts_to_plot_np.shape[1])) 
        )
        plt.savefig(output_filename)
        plt.close(fig) 
        logger.info(f"Saved plot for subject {subject_id}, frame {frame_num + 1} to {output_filename}")

    except Exception as e:
        logger.error(f"Error plotting frame {frame_num + 1} for subject {subject_id} to {output_filename}: {e}", exc_info=True)

def plot_custom_sensor_segmentation(
    frame_sensor_data_dt_idx: pd.DataFrame,
    frame_activity_labels_dt_idx: pd.Series, # Already time-shifted
    detected_cps_relative_indices: List[int], 
    gt_cps_relative_indices: List[int],
    subject_id: str,
    modality_name_for_plot: str,
    frame_num: int, # 0-indexed
    output_dir: str,
    sampling_rate: float, # Used for converting CP indices to time if needed, but here CPs are indices
    unknown_activity_label: str = "Unknown" 
):
    """
    Generates and saves a custom plot with sensor data, shaded activities, and changepoints.
    Each sensor channel is plotted in a separate subplot.
    Activity labels are expected to have their DatetimeIndex already shifted.
    Changepoints are provided as integer indices relative to the start of the frame_sensor_data_dt_idx.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if frame_sensor_data_dt_idx.empty:
        logger.warning(f"Custom plot: Frame {frame_num + 1} for {subject_id} sensor data is empty. Skipping plot.")
        return

    channels = frame_sensor_data_dt_idx.columns
    num_channels = len(channels)
    if num_channels == 0:
        logger.warning(f"Custom plot: Frame {frame_num + 1} for {subject_id} has no sensor data channels. Skipping plot.")
        return

    fig, axes = plt.subplots(num_channels, 1, figsize=(20, 3 * num_channels), sharex=True, squeeze=False)
    axes = axes.flatten() # Ensure axes is always a 1D array

    # Plot sensor data and CPs for each channel
    for i, channel in enumerate(channels):
        ax = axes[i]
        ax.plot(frame_sensor_data_dt_idx.index, frame_sensor_data_dt_idx[channel], label=channel, color='black', linewidth=0.8)
        ax.set_ylabel(channel)
        if i == num_channels - 1:
            ax.set_xlabel("Time")
        
        # Plot detected changepoints
        for cp_idx in detected_cps_relative_indices:
            if 0 <= cp_idx < len(frame_sensor_data_dt_idx):
                cp_time = frame_sensor_data_dt_idx.index[cp_idx]
                ax.axvline(cp_time, color='red', linestyle='--', linewidth=1.2, label='Detected CP' if i == 0 and cp_idx == detected_cps_relative_indices[0] else None)

        # Plot ground truth changepoints
        for gt_cp_idx in gt_cps_relative_indices:
            if 0 <= gt_cp_idx < len(frame_sensor_data_dt_idx):
                gt_cp_time = frame_sensor_data_dt_idx.index[gt_cp_idx]
                ax.axvline(gt_cp_time, color='green', linestyle=':', linewidth=1.2, label='Ground Truth CP' if i == 0 and gt_cp_idx == gt_cps_relative_indices[0] else None)

    # Define a color map for activities
    unique_activities_for_coloring = sorted(list(set(
        act for act in frame_activity_labels_dt_idx.unique() 
        if pd.notna(act) and act != unknown_activity_label
    )))
    
    activity_color_map = {}
    if unique_activities_for_coloring:
        try:
            cmap_name = 'tab10' if len(unique_activities_for_coloring) <= 10 else 'tab20' if len(unique_activities_for_coloring) <= 20 else 'viridis'
            if hasattr(plt, 'colormaps'): cmap = plt.colormaps.get_cmap(cmap_name)
            else: cmap = plt.cm.get_cmap(cmap_name)

            if cmap.N >= len(unique_activities_for_coloring): # Discrete colormap with enough colors
                colors_list = [cmap(c) for c in range(len(unique_activities_for_coloring))]
            else: # Sample from continuous colormap
                colors_list = [cmap(j / (len(unique_activities_for_coloring) -1)) if len(unique_activities_for_coloring) > 1 else cmap(0.5) for j in range(len(unique_activities_for_coloring))]
            activity_color_map = {activity: colors_list[j] for j, activity in enumerate(unique_activities_for_coloring)}
        except Exception as e_cmap:
            logger.warning(f"Colormap generation for activities failed: {e_cmap}. Using fallback colors.")
            fb_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            for j, activity in enumerate(unique_activities_for_coloring): activity_color_map[activity] = fb_colors[j % len(fb_colors)]
    
    default_shade_color = 'lightgrey'

    # Shade activity regions on all subplots
    if not frame_activity_labels_dt_idx.empty:
        current_activity_start_time = None
        current_activity = None
        
        # Ensure frame_activity_labels_dt_idx.index is a DatetimeIndex if frame_sensor_data_dt_idx.index is
        # This is important if sensor data has time index but activities were passed as simple series
        if isinstance(frame_sensor_data_dt_idx.index, pd.DatetimeIndex) and not isinstance(frame_activity_labels_dt_idx.index, pd.DatetimeIndex):
            if len(frame_activity_labels_dt_idx) == len(frame_sensor_data_dt_idx):
                frame_activity_labels_dt_idx.index = frame_sensor_data_dt_idx.index
            else:
                logger.warning(f"[{subject_id}] Frame {frame_num+1}: Length mismatch, cannot align activity index to sensor data time index for shading.")


        for k in range(len(frame_activity_labels_dt_idx)):
            timestamp = frame_activity_labels_dt_idx.index[k]
            activity = frame_activity_labels_dt_idx.iloc[k]

            if current_activity is None: # First data point
                current_activity_start_time = timestamp
                current_activity = activity
            
            if activity != current_activity: # Activity changed
                if current_activity_start_time is not None and current_activity != unknown_activity_label and pd.notna(current_activity):
                    end_time_of_segment = timestamp 
                    color_to_use = activity_color_map.get(current_activity, default_shade_color)
                    for ax_k in axes:
                        ax_k.axvspan(current_activity_start_time, end_time_of_segment, color=color_to_use, alpha=0.2, label=f'_{current_activity}_shade') # Internal label
                
                current_activity_start_time = timestamp 
                current_activity = activity

        # Shade the last activity segment
        if current_activity_start_time is not None and current_activity != unknown_activity_label and pd.notna(current_activity) and len(frame_activity_labels_dt_idx) > 0:
            # Determine the end time for the last segment carefully
            if isinstance(frame_sensor_data_dt_idx.index, pd.DatetimeIndex) and not frame_sensor_data_dt_idx.empty:
                 end_of_frame_time = frame_sensor_data_dt_idx.index[-1]
            elif isinstance(frame_activity_labels_dt_idx.index, pd.DatetimeIndex) and not frame_activity_labels_dt_idx.empty:
                 end_of_frame_time = frame_activity_labels_dt_idx.index[-1]
            else: # Fallback if no datetime index, use the last available timestamp if possible
                end_of_frame_time = timestamp # timestamp from the loop is the last activity's start time

            color_to_use = activity_color_map.get(current_activity, default_shade_color)
            for ax_k in axes:
                ax_k.axvspan(current_activity_start_time, end_of_frame_time, color=color_to_use, alpha=0.2, label=f'_{current_activity}_shade') # Internal label
    
    activities_in_frame_str = ", ".join(unique_activities_for_coloring) if unique_activities_for_coloring else "None"
    fig.suptitle(f"Custom Segmentation: {subject_id} - Frame {frame_num + 1}\\n"
                 f"Modality: {modality_name_for_plot} (Channels: {', '.join(channels)})\\n"
                 f"Shifted Activities: {activities_in_frame_str}", fontsize=12)
    
    # Consolidated Legend Creation
    final_legend_handles = []
    final_legend_labels = []
    
    # 1. Collect handles/labels from sensor data lines and CPs from all axes
    temp_handles_labels_map = {} # Use a map to ensure uniqueness of labels from lines/markers
    for ax_k in axes:
        h_ax, l_ax = ax_k.get_legend_handles_labels()
        for h_item, l_item in zip(h_ax, l_ax):
            if not l_item.startswith('_') and l_item not in temp_handles_labels_map: 
                temp_handles_labels_map[l_item] = h_item
    
    final_legend_handles.extend(temp_handles_labels_map.values())
    final_legend_labels.extend(temp_handles_labels_map.keys())

    # 2. Add proxy artists for activity shadings
    # unique_activities_for_coloring is defined earlier in the function
    for activity_name in unique_activities_for_coloring:
        if activity_name not in final_legend_labels: # Add only if not already present
            color = activity_color_map.get(activity_name, default_shade_color)
            patch = plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.2) # Match axvspan style
            final_legend_handles.append(patch)
            final_legend_labels.append(activity_name)
            
    if final_legend_handles:
        fig.legend(final_legend_handles, final_legend_labels, loc='upper right', bbox_to_anchor=(0.98, 0.93), fontsize='small')

    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout for suptitle
    
    output_filename = os.path.join(output_dir, f"{subject_id}_{modality_name_for_plot.replace('_','')}_frame{frame_num + 1}_custom.png")
    try:
        plt.savefig(output_filename)
        logger.info(f"Saved custom plot for subject {subject_id}, frame {frame_num + 1} to {output_filename}")
    except Exception as e:
        logger.error(f"Error saving custom plot for subject {subject_id}, frame {frame_num + 1} to {output_filename}: {e}", exc_info=True)
    finally:
        plt.close(fig)

# --- Main Processing Function ---
def run_segmentation_pipeline(main_cfg: Dict[str, Any], project_root_dir: str):
    """
    Main function to run the Claspy segmentation and plotting pipeline.
    """
    logger.info("--- Starting Claspy Multivariate Segmentation Pipeline ---")

    sync_params_data = load_sync_parameters(project_root_dir) # Load sync parameters

    processed_data_dir_name = main_cfg.get('processed_data_output_dir', 'processed_subjects') 
    processed_data_input_dir = os.path.join(project_root_dir, processed_data_dir_name)

    subjects_to_process = main_cfg.get('Stage 1', {}).get('subjects_to_load', [])
    if not subjects_to_process: 
        logger.warning("'subjects_to_load' not found or empty in config['Stage 1']. "
                       "Attempting to use 'subjects_to_process' from config['Stage 0'] if available, "
                       "otherwise, will try to list subjects from processed_data_input_dir.")
        subjects_to_process = main_cfg.get('subjects_to_process', []) 
        if not subjects_to_process:
            if os.path.exists(processed_data_input_dir):
                try:
                    subjects_to_process = [
                        s.replace('_filtered_corrected.pkl', '')
                        for s in os.listdir(processed_data_input_dir)
                        if s.endswith('_filtered_corrected.pkl')
                    ]
                    logger.info(f"Found {len(subjects_to_process)} subjects in {processed_data_input_dir}")
                except Exception as e:
                    logger.error(f"Could not list subjects from {processed_data_input_dir}: {e}")
                    subjects_to_process = []
            if not subjects_to_process:
                logger.error("No subjects specified or found to process. Exiting.")
                return

    if not subjects_to_process:
        logger.error("No subjects to process. Please check 'subjects_to_load' in config.yaml or the processed data directory.")
        return
    subjects_to_process = ['OutSense-498']
    logger.info(f"Will process subjects: {subjects_to_process}")

    main_output_dir = os.path.join(project_root_dir, BASE_PLOT_OUTPUT_DIR_NAME)
    os.makedirs(main_output_dir, exist_ok=True)
    logger.info(f"Main output directory set to: {main_output_dir}")

    sampling_rate = main_cfg.get('downsample_freq', DEFAULT_SAMPLING_RATE) 
    frame_duration_samples = int(FRAME_DURATION_HOURS * 60 * 60 * sampling_rate)
    logger.info(f"Target sampling rate: {sampling_rate} Hz. Frame duration: {FRAME_DURATION_HOURS}hr(s) = {frame_duration_samples} samples.")

    run_mode = main_cfg.get('run_mode', 'full_processing') # Get run_mode from config
    logger.info(f"Running in mode: {run_mode}")

    for subject_id in subjects_to_process:
        logger.info(f"--- Processing Subject: {subject_id} ---")

        subject_pkl_file = os.path.join(processed_data_input_dir, f"{subject_id}_filtered_corrected.pkl")
        
        subject_df_full = load_subject_data(subject_id, subject_pkl_file)
        if subject_df_full is None or subject_df_full.empty:
            logger.warning(f"No data loaded for subject {subject_id}. Skipping.")
            continue

        subject_output_dir = os.path.join(main_output_dir, subject_id) # Define subject_output_dir early for CSV path
        os.makedirs(subject_output_dir, exist_ok=True) # Ensure it exists

        # Instantiate Claspy model once per subject.
        # This instance will be used for segmentation (if not plot_only) and then for plotting.
        clasp_model_instance = BinaryClaSPSegmentation(**CLASPY_PARAMS)
        detected_changepoints = np.array([]) # Initialize

        if run_mode == 'plot_only':
            logger.info(f"[{subject_id}] Plot-only mode: Attempting to load existing changepoints.")
            cp_csv_path = os.path.join(subject_output_dir, f"{subject_id}_detected_changepoints.csv")
            try:
                if not os.path.exists(cp_csv_path):
                    logger.error(f"[{subject_id}] Changepoints file {cp_csv_path} not found for plot_only mode. Skipping subject.")
                    continue
                cp_df = pd.read_csv(cp_csv_path)
                if 'changepoint_index' not in cp_df.columns:
                    logger.error(f"[{subject_id}] 'changepoint_index' column not found in {cp_csv_path}. Skipping subject.")
                    continue
                detected_changepoints = np.array(cp_df['changepoint_index'].tolist())
                logger.info(f"[{subject_id}] Loaded {len(detected_changepoints)} detected changepoints from {cp_csv_path}")
            except Exception as e: # Catch other potential errors like empty file, pd.read_csv issues
                logger.error(f"[{subject_id}] Error loading changepoints from {cp_csv_path}: {e}. Skipping subject.")
                continue
        
        elif run_mode == 'full_processing':
            logger.info(f"[{subject_id}] Full processing mode: Running segmentation.")
            segmentation_df = select_data_columns(subject_df_full, SEGMENTATION_COLUMN_PREFIXES)
            if segmentation_df.empty:
                logger.warning(f"No data columns selected for segmentation for subject {subject_id} with prefixes {SEGMENTATION_COLUMN_PREFIXES}. Skipping segmentation part.")
                # detected_changepoints remains empty, plotting will show no detected CPs
            else:
                segmentation_np_array = segmentation_df.to_numpy()
                if segmentation_np_array.ndim == 1: 
                    segmentation_np_array = segmentation_np_array.reshape(-1,1)
                logger.info(f"Data for segmentation ({subject_id}) shape: {segmentation_np_array.shape}")

                all_detected_changepoints_subject = []
                num_samples_total = segmentation_np_array.shape[0]

                logger.info(f"Starting chunk-wise segmentation for {subject_id}. Total samples: {num_samples_total}, Chunk size: {frame_duration_samples} samples.")

                if num_samples_total == 0:
                    logger.warning(f"Segmentation data for {subject_id} is empty. No segmentation will be performed.")
                    # detected_changepoints remains empty
                elif num_samples_total < frame_duration_samples and num_samples_total > 0 : 
                    logger.info(f"Total data ({num_samples_total} samples) is less than one chunk ({frame_duration_samples} samples). Segmenting as a single chunk.")
                    try:
                        detected_changepoints = clasp_model_instance.fit_predict(segmentation_np_array)
                        logger.info(f"Claspy found {len(detected_changepoints)} changepoints for {subject_id} (single chunk).")
                    except Exception as e:
                        logger.error(f"Error during Claspy segmentation for {subject_id} (single chunk): {e}", exc_info=True)
                        detected_changepoints = np.array([]) 
                else: # num_samples_total >= frame_duration_samples
                    total_chunks = (num_samples_total + frame_duration_samples - 1) // frame_duration_samples
                    for i_chunk, chunk_start_idx in enumerate(range(0, num_samples_total, frame_duration_samples)):
                        chunk_end_idx = min(chunk_start_idx + frame_duration_samples, num_samples_total)
                        current_chunk_data = segmentation_np_array[chunk_start_idx:chunk_end_idx]

                        if current_chunk_data.shape[0] == 0: 
                            logger.debug(f"Skipping empty chunk {i_chunk + 1}/{total_chunks} for {subject_id}: samples {chunk_start_idx}-{chunk_end_idx}")
                            continue
                        
                        logger.info(f"Segmenting chunk {i_chunk + 1}/{total_chunks} for {subject_id}: samples {chunk_start_idx}-{chunk_end_idx} (Shape: {current_chunk_data.shape})...")
                        try:
                            chunk_changepoints = clasp_model_instance.fit_predict(current_chunk_data)
                            adjusted_chunk_cps = [cp + chunk_start_idx for cp in chunk_changepoints]
                            all_detected_changepoints_subject.extend(adjusted_chunk_cps)
                            logger.info(f"Found {len(chunk_changepoints)} CPs in chunk {i_chunk + 1}. Total CPs for subject {subject_id} so far (pre-unique): {len(all_detected_changepoints_subject)}")
                        except Exception as e:
                            logger.error(f"Error segmenting chunk {i_chunk + 1}/{total_chunks} (samples {chunk_start_idx}-{chunk_end_idx}) for {subject_id}: {e}", exc_info=True)
                    
                    if all_detected_changepoints_subject:
                        detected_changepoints = np.array(sorted(list(set(all_detected_changepoints_subject))))
                        logger.info(f"Claspy found {len(detected_changepoints)} unique changepoints for {subject_id} after chunk-wise segmentation.")
                    else:
                        logger.info(f"No changepoints found for {subject_id} after chunk-wise segmentation.")
                        detected_changepoints = np.array([]) # Ensure it's an empty array if no CPs found
            
            save_changepoints_to_csv(detected_changepoints, subject_id, subject_output_dir)
        
        else: # Unknown run_mode
            logger.error(f"Unknown run_mode: '{run_mode}'. Skipping segmentation and plotting for subject {subject_id}.")
            continue


        # --- Plotting Stage (common for both modes if CPs are available or loaded) ---
        # We proceed to plotting if either:
        # 1. run_mode is 'full_processing' (CPs were just calculated or are empty if segmentation failed/skipped)
        # 2. run_mode is 'plot_only' AND detected_changepoints were successfully loaded (even if empty)
        # The `continue` statements earlier handle cases where plot_only fails to load CPs.

        plotting_df = select_data_columns(subject_df_full, [PLOTTING_MODALITY_PREFIX]) 
        if plotting_df.empty:
            logger.warning(f"No data columns selected for plotting for subject {subject_id} with prefix {PLOTTING_MODALITY_PREFIX}. Skipping plots.")
        else:
            gt_changepoints = get_ground_truth_changepoints(subject_df_full, ACTIVITY_COLUMN_NAME, UNKNOWN_ACTIVITY_LABEL)
            activity_series_for_frames = subject_df_full.get(ACTIVITY_COLUMN_NAME)

            random_frames_indices = select_random_time_frames(
                df_length=len(subject_df_full),
                num_frames=NUM_RANDOM_FRAMES,
                frame_duration_samples=frame_duration_samples,
                activity_series=activity_series_for_frames,
                unknown_label=UNKNOWN_ACTIVITY_LABEL
            )

            if not random_frames_indices:
                logger.warning(f"No time frames selected for plotting for subject {subject_id}.")
            else:
                # Ensure detected_changepoints is a list for plot_segmented_frame
                cps_for_plotting = list(detected_changepoints) if isinstance(detected_changepoints, np.ndarray) else []
                
                for i, (start_idx, end_idx) in enumerate(random_frames_indices):
                    logger.info(f"Preparing custom plot for frame {i+1}/{len(random_frames_indices)} (Indices: {start_idx}-{end_idx}) for {subject_id}...")
                    
                    # Comment out the old plot_segmented_frame call
                    # plot_segmented_frame(
                    #     claspy_model=clasp_model_instance, 
                    #     full_ts_data_for_plotting=plotting_df, 
                    #     all_detected_cps=cps_for_plotting, 
                    #     all_gt_cps=gt_changepoints,
                    #     frame_start_idx=start_idx,
                    #     frame_end_idx=end_idx,
                    #     subject_id=subject_id,
                    #     plot_modality_name=PLOTTING_MODALITY_PREFIX,
                    #     frame_num=i,
                    #     output_dir=subject_output_dir,
                    #     activity_labels_full=activity_series_for_frames if activity_series_for_frames is not None else pd.Series(dtype='object'),
                    #     sampling_rate=sampling_rate
                    # )

                    # --- Custom Plot Call ---
                    # Data for current frame for plotting sensor values
                    plotting_df_frame = plotting_df.iloc[start_idx : end_idx + 1]
                    
                    # Activity labels are taken directly, assuming time shift is already handled in source data
                    frame_activity_labels_for_custom_plot = subject_df_full[ACTIVITY_COLUMN_NAME].iloc[start_idx : end_idx + 1]
                    
                    # Ensure plotting_df_frame has DatetimeIndex if subject_df_full has it
                    if isinstance(subject_df_full.index, pd.DatetimeIndex) and not isinstance(plotting_df_frame.index, pd.DatetimeIndex):
                        idx_slice = subject_df_full.index[start_idx : end_idx + 1]
                        # Ensure the slice length matches plotting_df_frame's length before assigning
                        if len(idx_slice) >= len(plotting_df_frame):
                            plotting_df_frame.index = idx_slice[:len(plotting_df_frame)]
                        else:
                            logger.warning(f"[{subject_id}] Frame {i+1}: Not enough index values from subject_df_full to assign to plotting_df_frame. Index assignment skipped.")

                    # Ensure frame_activity_labels_for_custom_plot has DatetimeIndex if subject_df_full has it
                    if isinstance(subject_df_full.index, pd.DatetimeIndex) and not isinstance(frame_activity_labels_for_custom_plot.index, pd.DatetimeIndex):
                        idx_slice = subject_df_full.index[start_idx : end_idx + 1]
                        # Ensure the slice length matches frame_activity_labels_for_custom_plot's length
                        if len(idx_slice) >= len(frame_activity_labels_for_custom_plot):
                           frame_activity_labels_for_custom_plot.index = idx_slice[:len(frame_activity_labels_for_custom_plot)]
                        else:
                            logger.warning(f"[{subject_id}] Frame {i+1}: Not enough index values from subject_df_full to assign to frame_activity_labels_for_custom_plot. Index assignment skipped.")
                    
                    # Relative changepoints for the current frame
                    current_detected_cps_list = []
                    if isinstance(detected_changepoints, np.ndarray):
                        current_detected_cps_list = detected_changepoints.tolist()
                    elif isinstance(detected_changepoints, list):
                        current_detected_cps_list = detected_changepoints
                    
                    # Ensure changepoints are integers before subtraction
                    frame_detected_cps_relative = [int(cp) - start_idx for cp in current_detected_cps_list if start_idx <= int(cp) < end_idx]
                    frame_gt_cps_relative = [int(cp) - start_idx for cp in gt_changepoints if start_idx <= int(cp) < end_idx]

                    plot_custom_sensor_segmentation(
                        frame_sensor_data_dt_idx=plotting_df_frame,
                        frame_activity_labels_dt_idx=frame_activity_labels_for_custom_plot,
                        detected_cps_relative_indices=frame_detected_cps_relative,
                        gt_cps_relative_indices=frame_gt_cps_relative,
                        subject_id=subject_id,
                        modality_name_for_plot=PLOTTING_MODALITY_PREFIX,
                        frame_num=i,
                        output_dir=subject_output_dir,
                        sampling_rate=sampling_rate,
                        unknown_activity_label=UNKNOWN_ACTIVITY_LABEL
                    )
        logger.info(f"Finished processing for subject {subject_id}.")
    logger.info("--- Claspy Multivariate Segmentation Pipeline Complete ---")

# --- if __name__ block ---
if __name__ == '__main__':
    logger.info("Script execution started directly.")
    
    project_root = get_project_root()
    logger.info(f"Project root determined as: {project_root}")

    try:
        main_config = load_main_config(project_root)
        
        run_segmentation_pipeline(main_config, project_root)

    except FileNotFoundError as e:
        logger.error(f"Configuration file error: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred in the main execution block: {e}", exc_info=True)
    
    logger.info("Script execution finished.")