import os
import yaml
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import ruptures as rpt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import seaborn as sns
import re
from sklearn.preprocessing import StandardScaler # For loading the scaler

# --- Configuration (User to modify these path/model specifics) ---
CONFIG_FILE = "config.yaml"
SYNC_PARAMS_FILE = "Sync_Parameters.yaml" # Assuming it's in the same directory or project root

TARGET_SUBJECT_ID = 'OutSense-498'  # Example: Set this to the subject you want to process
TARGET_DATE_HOUR_STR = None # Example: "YYYY-MM-DD HH", or None to process all subject data
# CHANNELS_TO_USE = ['wrist_acc_x'] # Example for single channel AE
CHANNELS_TO_USE = ['wrist_acc_x', 'wrist_acc_y', 'wrist_acc_z', 'x_axis_g', 'y_axis_g', 'z_axis_g', 'bioz_acc_x', 'bioz_acc_y', 'bioz_acc_z', 'x_axis_dps', 'y_axis_dps', 'x_axis_dps', 'vivalnk_acc_x', 'vivalnk_acc_y', 'vivalnk_acc_z', 'bottom_value_1', 'bottom_value_2', 'bottom_value_3', 'bottom_value_4', 'bottom_value_5', 'bottom_value_6', 'bottom_value_7', 'bottom_value_8', 'bottom_value_9', 'bottom_value_10', 'bottom_value_11'] # Example multivariate

AE_MODEL_TYPE = 'LSTM'  # Choose 'LSTM' or 'TCN'
# IMPORTANT: User needs to provide paths to their pre-trained models and scaler
PRETRAINED_MODEL_PATH_LSTM = "models/lstm_ae_all_sensors_OutSense-498.pth" # Placeholder
PRETRAINED_MODEL_PATH_TCN = "models/tcn_autoencoder.pth"   # Placeholder
SCALER_PATH = "models/scaler_all_sensors_OutSense-498.pkl" # Scaler fitted on AE training data

SEQUENCE_DURATION_SEC = 5
RECONSTRUCTION_ERROR_CPD_PENALTY = 10 # Example penalty for CPD on error signal
# For plotting multiple windows of the error signal
ERROR_PLOT_WINDOW_SIZE_POINTS = 72000 
NUM_ERROR_PLOT_WINDOWS = 5


# --- Autoencoder Model Definitions (Placeholders - these MUST match your trained models) ---
class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features, embedding_dim=128, hidden_dim=256, n_layers=4, dropout=0.1):
        super(LSTMAutoencoder, self).__init__()
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Encoder
        self.lstm1_enc = nn.LSTM(n_features, hidden_dim, n_layers, batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.dropout_enc = nn.Dropout(dropout)
        self.fc_enc = nn.Linear(hidden_dim, embedding_dim)
        
        # Decoder
        self.lstm1_dec = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.dropout_dec = nn.Dropout(dropout)
        self.fc_dec = nn.Linear(hidden_dim, n_features)
        
        print(f"LSTMAutoencoder: n_features={n_features}, embedding_dim={embedding_dim}, hidden_dim={hidden_dim}, n_layers={n_layers}, dropout={dropout}")


    def forward(self, x):
        # x shape: (batch_size, seq_len, n_features)
        _, (hn_enc, _) = self.lstm1_enc(x)
        encoded_hidden = self.dropout_enc(hn_enc[-1]) # Use last layer's hidden state
        encoded = self.fc_enc(encoded_hidden) # (batch_size, embedding_dim)
        
        seq_len = x.size(1)
        # Repeat encoded representation for each time step to feed into decoder LSTM
        decoded_input = encoded.unsqueeze(1).repeat(1, seq_len, 1) # (batch_size, seq_len, embedding_dim)
        
        out_dec, _ = self.lstm1_dec(decoded_input)
        out_dec_dropped = self.dropout_dec(out_dec)
        reconstructed = self.fc_dec(out_dec_dropped)
        return reconstructed

class TCNAutoencoder(nn.Module):
    # Basic TCN Autoencoder structure - User needs to define/import their actual TCN model
    # This is a simplified placeholder example. Real TCNs have more complex layered structures.
    def __init__(self, n_features, num_channels, kernel_size=3, dropout=0.2):
        super(TCNAutoencoder, self).__init__()
        # Placeholder: User should implement or import their TCN architecture
        # For simplicity, using a few conv layers as a stand-in conceptual TCN structure
        # Encoder
        self.conv1 = nn.Conv1d(n_features, num_channels[0], kernel_size, padding=(kernel_size - 1) // 2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(num_channels[0], num_channels[1], kernel_size, padding=(kernel_size - 1) // 2)
        self.relu2 = nn.ReLU()
        # Bottleneck (example: could be another conv to reduce channels further or a dense layer if flattening)
        self.bottleneck_conv = nn.Conv1d(num_channels[1], num_channels[1]//2, 1) # Example bottleneck

        # Decoder
        self.deconv1 = nn.ConvTranspose1d(num_channels[1]//2, num_channels[1], 1)
        self.relu3 = nn.ReLU()
        self.deconv2 = nn.ConvTranspose1d(num_channels[1], num_channels[0], kernel_size, padding=(kernel_size - 1) // 2)
        self.relu4 = nn.ReLU()
        self.deconv3 = nn.ConvTranspose1d(num_channels[0], n_features, kernel_size, padding=(kernel_size - 1) // 2)

        print(f"Placeholder TCNAutoencoder initialized. n_features={n_features}, num_channels={num_channels}")


    def forward(self, x):
        # x shape: (batch_size, seq_len, n_features) -> permute to (batch_size, n_features, seq_len) for Conv1D
        x = x.permute(0, 2, 1)
        # Encode
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        encoded = self.bottleneck_conv(x) # Bottleneck
        # Decode
        x = self.relu3(self.deconv1(encoded))
        x = self.relu4(self.deconv2(x))
        reconstructed = self.deconv3(x)
        # Permute back: (batch_size, n_features, seq_len) -> (batch_size, seq_len, n_features)
        reconstructed = reconstructed.permute(0, 2, 1)
        return reconstructed

# --- Helper Functions ---
def parse_label_time_shift(shift_str: str) -> pd.Timedelta:
    """
    Parses a time shift string like "-1h 0min 15s" into a pd.Timedelta object.
    Relies on pd.to_timedelta's ability to parse this format directly.
    """
    if not shift_str or not isinstance(shift_str, str):
        print(f"Warning: Invalid shift_str input for parsing: '{shift_str}'. Returning zero shift.")
        return pd.Timedelta(seconds=0)
    
    try:
        # pd.to_timedelta can directly handle formats like "Xh Ymin Zs", 
        # e.g., "-1h 0min 15s" or "-2h -1min -25s", including those with spaces.
        # It also handles more verbose formats like "-1 hours 0 minutes 15 seconds".
        print(shift_str)
        delta = pd.to_timedelta(shift_str)
        print(delta)
        return delta
    except ValueError as e:
        # This block will catch errors if the string is not in a format pd.to_timedelta understands.
        print(f"Warning: Could not parse time shift string '{shift_str}' directly using pd.to_timedelta: {e}.")
        print(f"         Please ensure the format is like 'Xh Ymin Zs' (e.g., '-1h 0min 15s') or other pandas-compatible formats.")
        print(f"         The problematic string was: '{shift_str}'. Returning zero shift.")
        return pd.Timedelta(seconds=0)

def load_pipeline_data(config, project_root):
    intermediate_dir_config = config.get('intermediate_feature_dir', 'features')
    intermediate_dir = os.path.join(project_root, intermediate_dir_config)
    data_filename = "combined_cleaned_data.pkl"
    file_path = os.path.join(intermediate_dir, data_filename)
    if not os.path.exists(file_path):
        print(f"Error: Data file not found: {file_path}"); return pd.DataFrame()
    try:
        df = pd.read_pickle(file_path)
        print(f"Successfully loaded data '{file_path}'. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}"); return pd.DataFrame()

def create_sequences_with_timestamps(data_df_channels: pd.DataFrame, sequence_length_samples: int, step_size: int = 1):
    sequences = []
    sequence_start_timestamps = []
    values = data_df_channels.values
    index = data_df_channels.index

    for i in range(0, len(values) - sequence_length_samples + 1, step_size):
        sequences.append(values[i : i + sequence_length_samples])
        sequence_start_timestamps.append(index[i])
    
    if not sequences: # Handle case where data is shorter than sequence_length_samples
        return np.array([]), []

    return np.array(sequences, dtype=np.float32), sequence_start_timestamps


def get_reconstruction_error(model, sequences_np, device, batch_size=128):
    model.eval()
    model.to(device)
    errors = []
    num_sequences = sequences_np.shape[0]

    with torch.no_grad():
        for i in range(0, num_sequences, batch_size):
            batch_sequences = sequences_np[i : i + batch_size]
            batch_sequences_torch = torch.from_numpy(batch_sequences).to(device)
            
            reconstructed = model(batch_sequences_torch)
            
            # Calculate MSE for each sequence in the batch
            # (input - output)^2 -> mean over features -> mean over time steps
            batch_error = torch.mean(torch.mean((batch_sequences_torch - reconstructed)**2, axis=2), axis=1)
            errors.extend(batch_error.cpu().numpy())
            
    return np.array(errors, dtype=np.float32)


def detect_change_points_on_error_signal(error_series: pd.Series, model_type="pelt", pen_value=3):
    if error_series.empty: return []
    if error_series.isnull().any():
        print("Warning: Reconstruction error series contains NaN values. Filling with ffill/bfill.")
        error_series = error_series.fillna(method='ffill').fillna(method='bfill')
    if error_series.isnull().any(): # If still NaNs (e.g., all NaNs)
        print("Warning: Reconstruction error series still contains NaN values after fill. Cannot detect change points.")
        return []

    points = error_series.values.reshape(-1, 1)
    if model_type == "pelt":
        algo = rpt.Pelt(model="l2").fit(points)
    elif model_type == "binseg":
        algo = rpt.Binseg(model="l2").fit(points)
    else: # Default or unsupported
        print(f"Unsupported CPD model '{model_type}' for error signal. Defaulting to Pelt.")
        algo = rpt.Pelt(model="l2").fit(points)
    
    try:
        # .predict returns indices relative to the `points` array
        bkps_indices = algo.predict(pen=pen_value)
    except Exception as e:
        print(f"Error during change point detection on error signal: {e}"); return []
    return bkps_indices # These are integer indices

def visualize_error_segmentation(error_series, change_point_indices, 
                                 activity_labels_df, target_label_col_name, 
                                 plot_filename_suffix="", title_prefix="Reconstruction Error Segmentation",
                                 plot_label_colors_config=None):
    if error_series.empty:
        print(f"Skipping plot {plot_filename_suffix}: Error series is empty.")
        return

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(18, 7))
    
    # Plot reconstruction error
    ax.plot(error_series.index, error_series.values, label="Reconstruction Error", color='dodgerblue', linewidth=1.0)

    # Plot change points
    cp_label_added = False
    for cp_idx in change_point_indices:
        # cp_idx is an integer index for error_series
        if 0 <= cp_idx < len(error_series.index):
            timestamp_at_cp = error_series.index[cp_idx]
            ax.axvline(timestamp_at_cp, color='red', linestyle='--', linewidth=1.5, 
                       label='Change Point' if not cp_label_added else None)
            cp_label_added = True
        elif cp_idx == len(error_series.index) and not error_series.empty: # Ruptures way of marking end
             ax.axvline(error_series.index[-1], color='red', linestyle=':', linewidth=1.5, 
                       label='End of Data (CP)' if not cp_label_added else None) # Indicate it's the end
             cp_label_added = True


    # Overlay activity labels
    plotted_activities_for_legend = set()
    # Use a local color map to store colors for activities seen in this specific plot call
    # to handle random color generation consistently if needed.
    local_color_map_for_plot = {} 

    current_plot_start_time = error_series.index.min()
    current_plot_end_time = error_series.index.max()

    if activity_labels_df is not None and not activity_labels_df.empty and target_label_col_name in activity_labels_df.columns:
        overlapping_labels = activity_labels_df[
            (activity_labels_df['Real_Start_Time'] < current_plot_end_time) &
            (activity_labels_df['Real_End_Time'] > current_plot_start_time)
        ]
        if not overlapping_labels.empty:
            unique_activities_in_overlap = sorted(overlapping_labels[target_label_col_name].astype(str).unique())
            
            # Prepare colors: use config if available, then local_color_map, then random
            palette = sns.color_palette("Paired", n_colors=max(len(unique_activities_in_overlap),1))
            
            for i, activity_name_str in enumerate(unique_activities_in_overlap):
                color = None
                if plot_label_colors_config and activity_name_str in plot_label_colors_config:
                    color = plot_label_colors_config[activity_name_str]
                
                if color is None: # Not in config or config is None
                    color = palette[i % len(palette)] # Fallback to generated palette
                local_color_map_for_plot[activity_name_str] = color


            for _, row in overlapping_labels.iterrows():
                activity = str(row[target_label_col_name])
                start_time = pd.to_datetime(row['Real_Start_Time'])
                end_time = pd.to_datetime(row['Real_End_Time'])
                
                visible_start_time = max(start_time, current_plot_start_time)
                visible_end_time = min(end_time, current_plot_end_time)

                if visible_start_time < visible_end_time:
                    ax.axvspan(visible_start_time, visible_end_time, 
                               color=local_color_map_for_plot.get(activity, 'lightgray'), alpha=0.3, zorder=0)
                    plotted_activities_for_legend.add(activity)
    
    handles, labels = ax.get_legend_handles_labels()
    if plotted_activities_for_legend:
        for activity in sorted(list(plotted_activities_for_legend)):
            if str(activity) not in labels and f"Act: {str(activity)}" not in labels : # Avoid duplicate legend entries
                handles.append(mpatches.Patch(color=local_color_map_for_plot.get(activity, 'lightgray'), alpha=0.3))
                labels.append(f"Act: {str(activity)}") # Make activity labels distinct

    if handles: # Only show legend if there's something in it
        ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0.)

    title_str = title_prefix
    if not error_series.empty:
         title_str += f"\n(Period: {error_series.index.min().strftime('%H:%M:%S')} to {error_series.index.max().strftime('%H:%M:%S')} on {error_series.index.min().strftime('%Y-%m-%d')})"
    ax.set_title(title_str, fontsize=12)
    ax.set_xlabel("Time", fontsize=10)
    ax.set_ylabel("Reconstruction Error", fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.xticks(rotation=30, ha='right')
    plt.subplots_adjust(right=0.80, bottom=0.15)
    
    output_filename = f"ae_segmentation{plot_filename_suffix}.png"
    try:
        plt.savefig(output_filename)
        print(f"Saved AE segmentation visualization to {output_filename}")
    except Exception as e:
        print(f"Error saving plot {output_filename}: {e}")
    plt.close(fig)


# --- Main Function ---
def main():
    print(f"Starting AE-based Segmentation Script...")
    # 1. Load config
    if not os.path.exists(CONFIG_FILE):
        print(f"Error: Configuration file '{CONFIG_FILE}' not found. CWD: {os.getcwd()}"); return
    try:
        with open(CONFIG_FILE, 'r') as f: config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading '{CONFIG_FILE}': {e}"); return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = config.get('project_root_dir', os.path.abspath(os.path.join(script_dir, '..')))
    config['project_root_dir'] = project_root # Ensure it's in config for other functions

    # Load Sync Parameters for label shifting
    sync_params_path = os.path.join(project_root, SYNC_PARAMS_FILE)
    sync_params = {}
    if os.path.exists(sync_params_path):
        try:
            with open(sync_params_path, 'r') as f: sync_params = yaml.safe_load(f)
        except Exception as e: print(f"Warning: Error loading sync params '{sync_params_path}': {e}")
    else: print(f"Warning: Sync params file '{sync_params_path}' not found.")

    # 2. Load all_data_df
    all_data_df = load_pipeline_data(config, project_root)
    if all_data_df.empty: print("Failed to load main data. Exiting."); return

    subject_id_col = config.get('subject_id_column', 'SubjectID')
    target_label_col = 'Label' # Using 'Label' as a sensible default now
    downsample_freq = config.get('downsample_freq', 20) # Hz

    # 3. Filter for TARGET_SUBJECT_ID
    if subject_id_col not in all_data_df.columns:
        print(f"Error: Subject ID column '{subject_id_col}' not found in data."); return
    subject_data_df = all_data_df[all_data_df[subject_id_col] == TARGET_SUBJECT_ID].copy()
    if subject_data_df.empty:
        print(f"Error: No data found for subject '{TARGET_SUBJECT_ID}'."); return
    print(f"Filtered data for subject '{TARGET_SUBJECT_ID}'. Shape: {subject_data_df.shape}")

    # 4. Load and process global_labels_df for this subject
    subject_labels_df_processed = pd.DataFrame()
    global_labels_file_cfg = config.get('global_labels_file')
    if global_labels_file_cfg:
        global_labels_path = os.path.join(project_root, global_labels_file_cfg)
        if os.path.exists(global_labels_path):
            try:
                glb_df = pd.read_csv(global_labels_path)
                subj_lbl_df = glb_df[glb_df['Video_File'].astype(str).str.contains(TARGET_SUBJECT_ID, na=False)].copy()
                if not subj_lbl_df.empty:
                    subj_lbl_df['Real_Start_Time'] = pd.to_datetime(subj_lbl_df['Real_Start_Time'], errors='coerce')
                    subj_lbl_df['Real_End_Time'] = pd.to_datetime(subj_lbl_df['Real_End_Time'], errors='coerce')
                    
                    shift_str = sync_params.get(TARGET_SUBJECT_ID, {}).get('Label_Time_Shift', "0s")
                    label_time_shift_delta = parse_label_time_shift(shift_str)
                    if label_time_shift_delta != pd.Timedelta(seconds=0):
                        subj_lbl_df['Real_Start_Time'] += label_time_shift_delta
                        subj_lbl_df['Real_End_Time'] += label_time_shift_delta
                    
                    subj_lbl_df.dropna(subset=['Real_Start_Time', 'Real_End_Time', target_label_col], inplace=True)
                    subject_labels_df_processed = subj_lbl_df
                    print(f"Processed {len(subject_labels_df_processed)} labels for subject '{TARGET_SUBJECT_ID}'.")
            except Exception as e: print(f"Error processing labels for {TARGET_SUBJECT_ID}: {e}")
        else: print(f"Warning: Global labels file not found: {global_labels_path}")
    else: print("Warning: 'global_labels_file' not specified in config.")


    # 5. Further filter subject_data_df by a specific hour if TARGET_DATE_HOUR_STR is set
    #    Also filter subject_labels_df_processed to match this hour for plotting.
    data_to_process = subject_data_df # Start with all subject data post-ID filter
    labels_for_plotting_period = subject_labels_df_processed # Labels corresponding to data_to_process

    if TARGET_DATE_HOUR_STR and isinstance(data_to_process.index, pd.DatetimeIndex):
        try:
            start_hour = pd.to_datetime(TARGET_DATE_HOUR_STR, format='%Y-%m-%d %H')
            end_hour = start_hour + pd.Timedelta(hours=1)
            print(f"Filtering data and labels for hour: {start_hour} to < {end_hour}")
            
            data_in_hour = data_to_process[(data_to_process.index >= start_hour) & (data_to_process.index < end_hour)]
            if data_in_hour.empty:
                print(f"No data found for subject '{TARGET_SUBJECT_ID}' in hour '{TARGET_DATE_HOUR_STR}'. Exiting."); return
            data_to_process = data_in_hour
            
            if not subject_labels_df_processed.empty:
                labels_for_plotting_period = subject_labels_df_processed[
                    (subject_labels_df_processed['Real_Start_Time'] < end_hour) &
                    (subject_labels_df_processed['Real_End_Time'] > start_hour)
                ].copy()
            print(f"Data shape for target hour: {data_to_process.shape}. {len(labels_for_plotting_period)} labels in this hour.")
        except ValueError:
            print(f"Invalid TARGET_DATE_HOUR_STR format. Use 'YYYY-MM-DD HH'. Processing all subject data."); # Fallback to all data for subject
    elif not isinstance(data_to_process.index, pd.DatetimeIndex):
        print("Warning: Data does not have DatetimeIndex. Cannot filter by hour. Processing all subject data.")


    # 6. Select CHANNELS_TO_USE
    missing_channels = [ch for ch in CHANNELS_TO_USE if ch not in data_to_process.columns]
    if missing_channels:
        print(f"Error: Channels {missing_channels} not found in data for subject '{TARGET_SUBJECT_ID}'. Available: {data_to_process.columns.tolist()}"); return
    
    selected_channel_data_df = data_to_process[CHANNELS_TO_USE].copy()
    if selected_channel_data_df.isnull().values.any(): # Check for NaNs before scaling
        print("Warning: NaNs found in selected channels. Applying ffill/bfill before scaling.")
        selected_channel_data_df.fillna(method='ffill', inplace=True)
        selected_channel_data_df.fillna(method='bfill', inplace=True)
        if selected_channel_data_df.isnull().values.any(): # If still NaNs, fill with 0
            print("Warning: NaNs still present after ffill/bfill. Filling with 0.")
            selected_channel_data_df.fillna(0, inplace=True)


    # ** SCALING STEP - Crucial for Autoencoders **
    # Load the scaler that was FIT ON THE AE TRAINING DATA
    if not os.path.exists(SCALER_PATH):
        print(f"Error: Scaler file '{SCALER_PATH}' not found. Cannot proceed without a pre-fitted scaler."); return
    try:
        scaler = pd.read_pickle(SCALER_PATH) # Assumes scaler was saved with pickle
        print(f"Applying pre-fitted scaler from {SCALER_PATH}")
        # Scaler expects numpy array (n_samples, n_features)
        # Ensure columns are in the same order as when scaler was fitted if scaler remembers feature names
        # For safety, if scaler was fitted on specific columns, ensure CHANNELS_TO_USE match.
        # Here, we assume scaler was fitted on data with columns in the order of CHANNELS_TO_USE.
        scaled_values = scaler.transform(selected_channel_data_df.values)
        scaled_channel_data_df = pd.DataFrame(scaled_values, index=selected_channel_data_df.index, columns=selected_channel_data_df.columns)
    except Exception as e:
        print(f"Error applying scaler: {e}"); return
    
    # 7. Determine sequence_length_samples
    sequence_length_samples = int(SEQUENCE_DURATION_SEC * downsample_freq)
    print(f"Using sequence length: {sequence_length_samples} samples ({SEQUENCE_DURATION_SEC}s at {downsample_freq}Hz)")

    # 8. Create sequences - using a step_size of 1 for a dense error signal
    print("Creating sequences for AE...")
    # Using the SCALED data to create sequences
    sequences_np, sequence_start_timestamps = create_sequences_with_timestamps(
        scaled_channel_data_df, 
        sequence_length_samples, 
        step_size=1 # Dense error signal
    )
    if sequences_np.shape[0] == 0:
        print("No sequences created. Data might be too short for the specified sequence length. Exiting."); return
    print(f"Created {sequences_np.shape[0]} sequences of shape {sequences_np.shape[1:]}")

    # 9. Load pre-trained AE model
    num_features = sequences_np.shape[2] # Should match len(CHANNELS_TO_USE)
    model_path = ""
    if AE_MODEL_TYPE == 'LSTM':
        model = LSTMAutoencoder(n_features=num_features) # Use actual params from training
        model_path = os.path.join(project_root, PRETRAINED_MODEL_PATH_LSTM)
    elif AE_MODEL_TYPE == 'TCN':
        # User needs to ensure num_channels list is appropriate for their trained TCN model
        # This is a placeholder, replace with actual TCN parameters from training
        num_tcn_channels = [32, 16, 8] # Example, should match TCN config
        if num_features == 1: # If single channel, TCN architecture might be simpler
            num_tcn_channels = [16, 8, 4] if len(CHANNELS_TO_USE)==1 else [min(32, num_features*2), min(16,num_features), min(8, num_features//2 if num_features > 1 else 1)]
            num_tcn_channels = [c for c in num_tcn_channels if c > 0]


        model = TCNAutoencoder(n_features=num_features, num_channels=num_tcn_channels, kernel_size=3)
        model_path = os.path.join(project_root, PRETRAINED_MODEL_PATH_TCN)
    else:
        print(f"Error: Unknown AE_MODEL_TYPE '{AE_MODEL_TYPE}'. Choose 'LSTM' or 'TCN'."); return

    if not os.path.exists(model_path):
        print(f"Error: Pre-trained model not found at '{model_path}'."); return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading pre-trained {AE_MODEL_TYPE} model from: {model_path} onto {device}")
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval() # Set to evaluation mode
    except Exception as e:
        print(f"Error loading model weights: {e}. Ensure model definition matches saved state_dict."); return

    # 10. Generate reconstruction errors
    print("Generating reconstruction errors...")
    reconstruction_errors_np = get_reconstruction_error(model, sequences_np, device)
    if len(reconstruction_errors_np) != len(sequence_start_timestamps):
        print("Error: Mismatch between number of errors and timestamps. Cannot create error series."); return

    reconstruction_error_series = pd.Series(reconstruction_errors_np, index=pd.DatetimeIndex(sequence_start_timestamps))
    print(f"Generated reconstruction error series. Length: {len(reconstruction_error_series)}")

    # 11. Visualize error signal, change points, and labels (in windows)
    # Change point detection will now be done PER WINDOW or for the full signal if plotted as one.
    N_error = len(reconstruction_error_series)
    fixed_num_plots = 5 # Generate 5 equally spaced plots

    if N_error == 0:
        print("Reconstruction error series is empty. No plots to generate.")
    elif N_error <= ERROR_PLOT_WINDOW_SIZE_POINTS:
        # If data is too short for even one standard window, or just fits one, plot all of it.
        print(f"Error signal length ({N_error}) is less than or equal to window size ({ERROR_PLOT_WINDOW_SIZE_POINTS}). Plotting full error signal.")
        
        # Perform CPD on the full error signal for this single plot
        print(f"Detecting change points on full error signal (length {N_error}) with penalty={RECONSTRUCTION_ERROR_CPD_PENALTY}...")
        cp_indices_for_plot = detect_change_points_on_error_signal(reconstruction_error_series, pen_value=RECONSTRUCTION_ERROR_CPD_PENALTY)
        if not cp_indices_for_plot:
            print("No change points detected in the reconstruction error signal.")
        else:
            # Log detected change points for this plot
            change_point_timestamps_plot = [reconstruction_error_series.index[i] for i in cp_indices_for_plot if i < len(reconstruction_error_series.index)]
            print(f"Detected {len(change_point_timestamps_plot)} change points for this plot.")
            if len(change_point_timestamps_plot) > 5:
                print(f"Example change point timestamps: {pd.to_datetime(change_point_timestamps_plot[:3]).strftime('%H:%M:%S').tolist()} ... {pd.to_datetime(change_point_timestamps_plot[-3:]).strftime('%H:%M:%S').tolist()}")
            else:
                print(f"Change point timestamps: {pd.to_datetime(change_point_timestamps_plot).strftime('%H:%M:%S').tolist()}")

        visualize_error_segmentation(
            reconstruction_error_series,
            cp_indices_for_plot, # Pass indices relative to reconstruction_error_series
            labels_for_plotting_period,
            target_label_col,
            plot_filename_suffix=f"_{TARGET_SUBJECT_ID}_{AE_MODEL_TYPE}_full_error_sig",
            title_prefix=f"AE Error Seg - {TARGET_SUBJECT_ID} - {AE_MODEL_TYPE}",
            plot_label_colors_config=config.get('plot_label_colors')
        )
    else:
        # N_error > ERROR_PLOT_WINDOW_SIZE_POINTS, so we can generate multiple window plots
        print(f"Generating {fixed_num_plots} equally spaced window plots for the reconstruction error signal...")
        
        for i in range(fixed_num_plots):
            start_idx_err = 0
            if fixed_num_plots > 1:
                # Calculate start index for equally spaced (potentially overlapping) windows
                start_idx_err = int(i * (N_error - ERROR_PLOT_WINDOW_SIZE_POINTS) / (fixed_num_plots - 1))
            
            start_idx_err = int(max(0, min(start_idx_err, N_error - ERROR_PLOT_WINDOW_SIZE_POINTS))) # Ensure start_idx is valid
            end_idx_err = min(start_idx_err + ERROR_PLOT_WINDOW_SIZE_POINTS, N_error)

            if start_idx_err >= end_idx_err:
                print(f"Skipping window {i+1} due to invalid range: start {start_idx_err}, end {end_idx_err}")
                continue

            error_series_window = reconstruction_error_series.iloc[start_idx_err:end_idx_err]
            if error_series_window.empty:
                print(f"Skipping window {i+1} as it resulted in an empty error series.")
                continue
            
            print(f"\\nProcessing plot window {i+1}/{fixed_num_plots} (Indices: {start_idx_err}-{end_idx_err-1} from original error series, Length: {len(error_series_window)})")
            
            # Perform CPD ON THIS WINDOW
            print(f"Detecting change points on error signal window with penalty={RECONSTRUCTION_ERROR_CPD_PENALTY}...")
            cp_indices_for_this_window_plot = detect_change_points_on_error_signal(error_series_window, pen_value=RECONSTRUCTION_ERROR_CPD_PENALTY)
            
            if not cp_indices_for_this_window_plot:
                print("No change points detected in this window.")
            else:
                # These indices are relative to error_series_window
                change_point_timestamps_window = [error_series_window.index[k] for k in cp_indices_for_this_window_plot if k < len(error_series_window.index)]
                print(f"Detected {len(cp_indices_for_this_window_plot)} change points in this window.")
                if len(change_point_timestamps_window) > 5:
                     print(f"Example change point timestamps in window: {pd.to_datetime(change_point_timestamps_window[:3]).strftime('%H:%M:%S').tolist()} ... {pd.to_datetime(change_point_timestamps_window[-3:]).strftime('%H:%M:%S').tolist()}")
                else:
                     print(f"Change point timestamps in window: {pd.to_datetime(change_point_timestamps_window).strftime('%H:%M:%S').tolist()}")

            plot_suffix = f"_{TARGET_SUBJECT_ID}_{AE_MODEL_TYPE}_err_win_{i+1}_of_{fixed_num_plots}"
            title = f"AE Error Seg - {TARGET_SUBJECT_ID} - {AE_MODEL_TYPE} (Window {i+1} of {fixed_num_plots})"

            visualize_error_segmentation(
                error_series_window,
                cp_indices_for_this_window_plot, # These are now directly from CPD on the window
                labels_for_plotting_period,      # visualize_error_segmentation filters labels based on error_series_window's time range
                target_label_col,
                plot_filename_suffix=plot_suffix,
                title_prefix=title,
                plot_label_colors_config=config.get('plot_label_colors')
            )
            
    print("--- AE Segmentation Script Finished ---")

if __name__ == "__main__":
    main()