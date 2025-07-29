# train_autoencoder.py

import os
import yaml
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# --- Configuration (Modify these as needed or move to a config file/args) ---
CONFIG_FILE = "config.yaml" # Path to your main pipeline config

# --- Autoencoder Model Definitions (These MUST match definitions in ae_segmentation.py) ---
class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features, embedding_dim=64, hidden_dim=32, n_layers=1, dropout=0.1):
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

class TCNLayer(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TCNLayer, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride,
                               padding=padding, dilation=dilation)
        self.chomp1 = nn.ConstantPad1d((0, -padding), 0) # Chomp to maintain causality/sequence length
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride,
                               padding=padding, dilation=dilation)
        self.chomp2 = nn.ConstantPad1d((0, -padding), 0)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        # x expected shape: (batch, channels, seq_len)
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCNAutoencoder(nn.Module):
    def __init__(self, n_features, num_channels_list, kernel_size=3, dropout=0.2):
        super(TCNAutoencoder, self).__init__()
        self.n_features = n_features
        
        # Encoder
        encoder_layers = []
        in_channels_enc = n_features
        for i, out_channels_enc in enumerate(num_channels_list):
            dilation_size = 2**i
            padding = (kernel_size - 1) * dilation_size # Standard padding for TCNs
            encoder_layers.append(TCNLayer(in_channels_enc, out_channels_enc, kernel_size, 
                                           stride=1, dilation=dilation_size, 
                                           padding=padding, dropout=dropout))
            in_channels_enc = out_channels_enc
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder (roughly mirrors encoder but with ConvTranspose1d or upsampling+conv)
        # For simplicity here, let's use Conv1d and assume TCN layers handle sequence length.
        # A more typical TCN decoder might use ConvTranspose1d.
        # This decoder part is a simplification and might need more sophisticated upsampling.
        decoder_layers = []
        # Last channel count from encoder is in_channels_enc
        # Reverse the num_channels_list for decoder, ending in n_features
        reversed_channels = num_channels_list[::-1]
        # Input to decoder is the last out_channels_enc
        # Output of first decoder layer should be the second to last encoder out_channels_enc

        # Example simplified decoder:
        # The last encoder layer output channels is num_channels_list[-1]
        # We need to go from num_channels_list[-1] back to n_features
        
        # Let's use a simpler decoder for now with Conv1d, assuming the TCN layers preserve length
        # This part would need careful design for a proper TCN AE.
        # Using a few simple Conv1d layers for conceptual symmetry
        
        # For a more robust TCN AE, decoder might use ConvTranspose1d or upsample + conv
        # This is a placeholder decoder structure:
        # It should mirror the encoder in reverse channel order
        # The last layer of the encoder has num_channels_list[-1] output channels.
        
        # Simplified decoder for placeholder
        self.decoder_fc1 = nn.Conv1d(num_channels_list[-1], num_channels_list[0], kernel_size=1) # To an intermediate
        self.decoder_final = nn.Conv1d(num_channels_list[0], n_features, kernel_size=1) # Back to original features


        print(f"TCNAutoencoder: n_features={n_features}, num_channels_list={num_channels_list}, kernel_size={kernel_size}, dropout={dropout}")
        print("Note: TCN decoder used here is a simplified placeholder.")


    def forward(self, x):
        # Input x shape: (batch_size, seq_len, n_features)
        x = x.permute(0, 2, 1)  # (batch_size, n_features, seq_len) for Conv1D TCN layers
        
        encoded = self.encoder(x)
        
        # Simplified decoder part
        decoded_intermediate = torch.relu(self.decoder_fc1(encoded))
        reconstructed = self.decoder_final(decoded_intermediate)

        reconstructed = reconstructed.permute(0, 2, 1)  # (batch_size, seq_len, n_features)
        return reconstructed


# --- Helper Functions ---
def load_and_prepare_data_for_ae_training(config, project_root, train_subject_ids, 
                                          channels_to_use, sequence_length_samples, 
                                          scaler_save_path):
    """
    Loads data for training subjects, selects channels, scales, and creates sequences.
    Saves the fitted scaler.
    """
    print("Loading and preparing data for AE training...")
    intermediate_dir_config = config.get('intermediate_feature_dir', 'features')
    intermediate_dir = os.path.join(project_root, intermediate_dir_config)
    data_filename = "combined_cleaned_data.pkl"
    file_path = os.path.join(intermediate_dir, data_filename)

    if not os.path.exists(file_path):
        print(f"Error: Data file not found: {file_path}"); return None, None
    try:
        all_data_df = pd.read_pickle(file_path)
        print(f"Loaded '{data_filename}'. Shape: {all_data_df.shape}")
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}"); return None, None

    subject_id_col = config.get('subject_id_column', 'SubjectID')
    if subject_id_col not in all_data_df.columns:
        print(f"Error: Subject ID column '{subject_id_col}' not found."); return None, None

    train_data_list = []
    for subj_id in train_subject_ids:
        subj_df = all_data_df[all_data_df[subject_id_col] == subj_id]
        if not subj_df.empty:
            train_data_list.append(subj_df[channels_to_use])
    
    if not train_data_list:
        print(f"No data found for training subjects: {train_subject_ids}."); return None, None
        
    train_df_combined = pd.concat(train_data_list, ignore_index=True) # Ignores datetime index for scaling
    print(f"Combined data for {len(train_subject_ids)} training subjects. Shape: {train_df_combined.shape}")

    # Handle NaNs before scaling
    if train_df_combined.isnull().values.any():
        print("Warning: NaNs found in training data before scaling. Applying ffill/bfill.")
        train_df_combined.fillna(method='ffill', inplace=True)
        train_df_combined.fillna(method='bfill', inplace=True)
        if train_df_combined.isnull().values.any(): # Should not happen if ffill/bfill worked
            print("Critical Warning: NaNs still present after ffill/bfill. Filling with 0, but check data.")
            train_df_combined.fillna(0, inplace=True)


    # Scale data
    scaler = StandardScaler()
    print(f"Fitting StandardScaler on {len(channels_to_use)} channels...")
    scaled_train_values = scaler.fit_transform(train_df_combined.values) # Fit on all time points of selected channels from training subjects
    
    # Save the scaler
    os.makedirs(os.path.dirname(scaler_save_path), exist_ok=True)
    with open(scaler_save_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to: {scaler_save_path}")

    # Reconstruct DataFrame with scaled values to pass to sequence creation (maintaining original structure for sequencing per subject)
    # This part is tricky because scaling was done on concatenated data. We need to apply scaling subject by subject if we want to preserve individual subject sequences
    # Alternative: scale subject by subject data for sequencing using the *globally fitted* scaler.

    all_sequences_list = []
    print("Creating sequences from scaled data for each training subject...")
    for subj_id in train_subject_ids: # Iterate again to get subject-specific data for sequencing
        subj_df = all_data_df[all_data_df[subject_id_col] == subj_id][channels_to_use].copy()
        if subj_df.empty: continue

        # Handle NaNs for this subject's data before scaling (consistency)
        if subj_df.isnull().values.any():
            subj_df.fillna(method='ffill', inplace=True)
            subj_df.fillna(method='bfill', inplace=True)
            subj_df.fillna(0, inplace=True) # Ensure no NaNs before transform

        # Transform this subject's data using the *already fitted* scaler
        subj_scaled_values = scaler.transform(subj_df.values)
        subj_scaled_df = pd.DataFrame(subj_scaled_values, index=subj_df.index, columns=subj_df.columns)

        subj_sequences = []
        for i in range(0, len(subj_scaled_df) - sequence_length_samples + 1, 1): # Step size 1 for overlap
            subj_sequences.append(subj_scaled_df.iloc[i : i + sequence_length_samples].values)
        
        if subj_sequences:
            all_sequences_list.extend(subj_sequences)

        print(f"Subject {subj_id}: Created {len(subj_sequences)} sequences of length {sequence_length_samples}.")

    if not all_sequences_list:
        print("No sequences created. Data might be too short or issues with subject filtering."); return None, scaler
        
    all_sequences_np = np.array(all_sequences_list, dtype=np.float32)
    print(f"Total sequences created for training: {all_sequences_np.shape}") # (num_sequences, seq_len, n_features)
    return all_sequences_np, scaler


def train_ae_model(model, train_sequences_np, learning_rate, batch_size, num_epochs, device, model_save_path):
    print(f"Starting training for model: {model.__class__.__name__}")
    model.to(device)
    
    dataset = TensorDataset(torch.from_numpy(train_sequences_np)) # Autoencoder target is input itself
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for i, (batch_sequences,) in enumerate(dataloader): # Note: (batch_sequences,)
            sequences = batch_sequences.to(device)
            
            optimizer.zero_grad()
            reconstructed = model(sequences)
            loss = criterion(reconstructed, sequences) # Compare input to output
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if (i + 1) % 100 == 0: # Log every 100 batches
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(dataloader)}], Loss: {loss.item():.6f}")

        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] completed. Average Training Loss: {avg_epoch_loss:.6f}")

    # Save the trained model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"Trained model saved to: {model_save_path}")


# --- Main Execution ---
if __name__ == "__main__":
    # --- Script Configuration ---
    AE_MODEL_TO_TRAIN = 'LSTM'  # 'LSTM' or 'TCN'
    # Define channels based on whether you're training a univariate or multivariate AE
    # CHANNELS_FOR_AE = ['wrist_acc_x'] # Example for univariate on one channel
    CHANNELS_FOR_AE = ['wrist_acc_x', 'wrist_acc_y', 'wrist_acc_z', 'x_axis_g', 'y_axis_g', 'z_axis_g', 'bioz_acc_x', 'bioz_acc_y', 'bioz_acc_z', 'x_axis_dps', 'y_axis_dps', 'x_axis_dps', 'vivalnk_acc_x', 'vivalnk_acc_y', 'vivalnk_acc_z', 'bottom_value_1', 'bottom_value_2', 'bottom_value_3', 'bottom_value_4', 'bottom_value_5', 'bottom_value_6', 'bottom_value_7', 'bottom_value_8', 'bottom_value_9', 'bottom_value_10', 'bottom_value_11'] # Example multivariate
    
    SEQUENCE_DURATION_SEC_TRAIN = 5 # Same as specified by user for segmentation script
    
    # AE Hyperparameters (examples, tune these)
    LSTM_EMBEDDING_DIM = 128
    LSTM_HIDDEN_DIM = 256
    LSTM_N_LAYERS = 4
    LSTM_DROPOUT = 0.1

    # For TCN, num_channels_list defines channels in each layer of encoder.
    # E.g., [64, 32, 16] means 3 layers in encoder, last one outputs 16 channels.
    # The n_features (input channels) is determined by CHANNELS_FOR_AE.
    TCN_NUM_CHANNELS_LIST = [len(CHANNELS_FOR_AE)*2, len(CHANNELS_FOR_AE), len(CHANNELS_FOR_AE)//2 if len(CHANNELS_FOR_AE)>1 else 1]
    TCN_NUM_CHANNELS_LIST = [c for c in TCN_NUM_CHANNELS_LIST if c > 0] # Ensure positive channels
    if not TCN_NUM_CHANNELS_LIST: TCN_NUM_CHANNELS_LIST = [len(CHANNELS_FOR_AE)] # Fallback if all became zero


    TCN_KERNEL_SIZE = 3
    TCN_DROPOUT = 0.2

    # Training Hyperparameters
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 1024 
    NUM_EPOCHS = 3

    # Output Paths (relative to project root, assuming this script is in src/ or similar)
    MODELS_OUTPUT_DIR = "models" # Will be project_root/models/
    SCALER_FILENAME = f"scaler_{'_'.join(CHANNELS_FOR_AE)}.pkl" # Scaler specific to channels used
    #MODEL_FILENAME_LSTM = f"lstm_ae_{'_'.join(CHANNELS_FOR_AE)}.pth"
    MODEL_FILENAME_TCN = f"tcn_ae_{'_'.join(CHANNELS_FOR_AE)}.pth"
    MODEL_FILENAME_LSTM = f"lstm_ae_all_sensors_OutSense-498.pth"
    SCALER_FILENAME = f"scaler_all_sensors_OutSense-498.pkl"
    print("--- Starting Autoencoder Training Script ---")
    # 1. Load config
    if not os.path.exists(CONFIG_FILE):
        print(f"Error: Configuration file '{CONFIG_FILE}' not found. CWD: {os.getcwd()}"); exit()
    try:
        with open(CONFIG_FILE, 'r') as f: config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading '{CONFIG_FILE}': {e}"); exit()

    script_main_dir = os.path.dirname(os.path.abspath(__file__))
    # Assume this script might be in 'src' or project root. Adjust if needed.
    # If script is in 'src', project_root is one level up.
    # If script is in project_root, project_root is current_dir.
    # Let's assume it's in src or a similar subfolder.
    project_root_default = os.path.abspath(os.path.join(script_main_dir, '..')) if "src" in script_main_dir.lower() else script_main_dir
    project_root = config.get('project_root_dir', project_root_default)
    
    print(f"Project root determined as: {project_root}")
    os.makedirs(os.path.join(project_root, MODELS_OUTPUT_DIR), exist_ok=True)
    
    scaler_save_path_full = os.path.join(project_root, MODELS_OUTPUT_DIR, SCALER_FILENAME)
    model_save_path_full_lstm = os.path.join(project_root, MODELS_OUTPUT_DIR, MODEL_FILENAME_LSTM)
    model_save_path_full_tcn = os.path.join(project_root, MODELS_OUTPUT_DIR, MODEL_FILENAME_TCN)

    # 2. Determine training subjects (all subjects MINUS test_subjects)
    all_subjects_in_config = config.get('subjects_to_load', []) # Or a more specific list of all available subjects
    test_subjects = config.get('test_subjects', [])
    train_subject_ids = [s for s in all_subjects_in_config if s not in test_subjects]
    if not train_subject_ids:
        print("Error: No training subjects found. Check 'subjects_to_load' and 'test_subjects' in config."); exit()
    print(f"Training AE on subjects: {train_subject_ids}")
    print(f"Using channels: {CHANNELS_FOR_AE}")

    # 3. Calculate sequence length in samples
    downsample_freq = config.get('downsample_freq', 20) # Default 20Hz
    sequence_length_samples_train = int(SEQUENCE_DURATION_SEC_TRAIN * downsample_freq)
    if sequence_length_samples_train <=0:
        print("Error: sequence_length_samples must be positive."); exit()

    # 4. Load and prepare data
    train_sequences, _ = load_and_prepare_data_for_ae_training(
        config, project_root, train_subject_ids, CHANNELS_FOR_AE, 
        sequence_length_samples_train, scaler_save_path_full
    )
    if train_sequences is None or train_sequences.shape[0] == 0:
        print("Failed to create training sequences. Exiting."); exit()

    # 5. Instantiate and train the chosen AE model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    n_features_for_ae = len(CHANNELS_FOR_AE) # Should match train_sequences.shape[2]

    if AE_MODEL_TO_TRAIN == 'LSTM':
        model_to_train = LSTMAutoencoder(
            n_features=n_features_for_ae,
            embedding_dim=LSTM_EMBEDDING_DIM,
            hidden_dim=LSTM_HIDDEN_DIM,
            n_layers=LSTM_N_LAYERS,
            dropout=LSTM_DROPOUT
        )
        model_final_save_path = model_save_path_full_lstm
    elif AE_MODEL_TO_TRAIN == 'TCN':
        model_to_train = TCNAutoencoder(
            n_features=n_features_for_ae,
            num_channels_list=TCN_NUM_CHANNELS_LIST, # Example: [n_features*2, n_features, n_features//2]
            kernel_size=TCN_KERNEL_SIZE,
            dropout=TCN_DROPOUT
        )
        model_final_save_path = model_save_path_full_tcn
    else:
        print(f"Error: Unknown AE_MODEL_TO_TRAIN '{AE_MODEL_TO_TRAIN}'. Choose 'LSTM' or 'TCN'."); exit()

    train_ae_model(model_to_train, train_sequences, LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS, device, model_final_save_path)
    
    print(f"--- Autoencoder Training Script Finished for {AE_MODEL_TO_TRAIN} ---")
    print(f"Model saved to: {model_final_save_path}")
    print(f"Scaler saved to: {scaler_save_path_full}")