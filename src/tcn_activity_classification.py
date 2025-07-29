# tcn_activity_classification.py

import os
import yaml
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import weight_norm
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight 
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

# --- Configuration (User to modify these) ---
CONFIG_FILE = "config.yaml"

# -- TCN Model Hyperparameters --
TCN_NUM_CHANNELS = [16, 128, 64, 64] 
TCN_KERNEL_SIZE = 3
TCN_DROPOUT = 0.16

# -- Training Hyperparameters --
LEARNING_RATE = 0.0004564931666269678
WEIGHT_DECAY = 0.00015622212464084883 # L2 regularization
BATCH_SIZE = 64
NUM_EPOCHS = 155 # Set a higher max number of epochs, as early stopping will find the best one
# -- Early Stopping Parameters --
EARLY_STOPPING_PATIENCE = 50 # Stop after 40 epochs with no improvement
EARLY_STOPPING_MIN_DELTA = 0.0001 # Minimum change to be considered an improvement
USE_EARLY_STOPPING = True # Default to True, will be overridden by config

# -- Monte Carlo Dropout Configuration --
MC_DROPOUT_SAMPLES = 100  # Number of forward passes for Monte Carlo Dropout. Set to 0 to disable.
                         # This can be moved to config.yaml if preferred for more dynamic configuration.

# -- Output Paths --
MODELS_OUTPUT_DIR = "models"
MODEL_FILENAME = f"tcn_classifier_best_model.pth" # Save the best model
LAST_MODEL_FILENAME = f"tcn_classifier_last_model.pth" # Save the last epoch model
SCALER_FILENAME = f"scaler_for_tcn.pkl"
ENCODER_FILENAME = f"label_encoder_for_tcn.pkl"

RANDOM_SEED = 42  # Set a random seed for reproducibility
np.random.seed(RANDOM_SEED)  # Set a random seed for reproducibility

# --- TCN Model Definition (No changes needed here) ---
class Chomp1d(nn.Module):
    def __init__(self, chomp_size): super(Chomp1d, self).__init__(); self.chomp_size = chomp_size
    def forward(self, x): return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()

        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding); self.relu1 = nn.ReLU(); self.dropout1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding); self.relu2 = nn.ReLU(); self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1, self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU(); self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01); self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None: self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x): out = self.net(x); res = x if self.downsample is None else self.downsample(x); return self.relu(out + res)

class TCNModel(nn.Module):
    def __init__(self, num_inputs, num_channels, num_classes, kernel_size=2, dropout=0.2):
        super(TCNModel, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        self.tcn = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], num_classes); self.init_weights()

    def init_weights(self): self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x): y = self.tcn(x); out = self.linear(y[:, :, -1]); return out

    def enable_dropout(self):
        """ Function to enable the dropout layers during inference. """
        for m in self.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train() # Enable dropout

# --- Data Preparation Function (MODIFIED to include validation set) ---
def prepare_data_for_tcn(config, project_root, plot_windows_enabled=False): 
    print("--- Starting Data Preparation for TCN (with Validation Set) ---")
    
    # 1. Load data (MODIFIED to load pre-windowed data from feature_engineering.py as .npy files)
    intermediate_dir = os.path.join(project_root, config.get('intermediate_feature_dir', 'features'))
    
    x_windows_path = os.path.join(intermediate_dir, "X_windows_raw.npy")
    print(f"INFO: Using raw windowed data from: {x_windows_path}")
        
    y_windows_path = os.path.join(intermediate_dir, "y_windows.npy")     # Changed from .pkl
    subject_ids_path = os.path.join(intermediate_dir, "subject_ids_windows.npy") # Changed from .pkl

    required_files = [x_windows_path, y_windows_path, subject_ids_path]
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"Error: Data file not found: {file_path}")
            print(f"Please ensure feature_engineering.py has been run and its outputs are in {intermediate_dir}")
            return [None]*8 

    try:
        # Load .npy files using np.load()
        X_windows = np.load(x_windows_path, allow_pickle=True) # allow_pickle=True if they might contain objects, though for X usually not needed
        y_windows = np.load(y_windows_path, allow_pickle=True) 
        subject_ids_windows = np.load(subject_ids_path, allow_pickle=True)
    except Exception as e:
        print(f"Error loading data from .npy files: {e}")
        return [None]*8

    # --- START LABEL REMAPPING ---
    print("INFO: Attempting to remap activity labels using Activity_Mapping_v2.csv")
    activity_mapping_path = os.path.join(project_root, "Activity_Mapping_v2.csv")
    if os.path.exists(activity_mapping_path):
        try:
            mapping_df = pd.read_csv(activity_mapping_path)
            # Create a dictionary for mapping: {Former_Label: New_Label}
            # Ensure column names match your CSV file exactly.
            # Assuming columns are 'Former_Label' and 'New_Label'
            if 'Former_Label' in mapping_df.columns and 'New_Label' in mapping_df.columns:
                label_mapping_dict = pd.Series(mapping_df.New_Label.values, index=mapping_df.Former_Label).to_dict()
                
                # Apply the mapping to y_windows
                # Vectorized mapping is faster: apply a function that looks up in dict, defaults to original if not found
                y_windows_original_for_debug = y_windows.copy() # Keep a copy for debugging if needed
                
                # Convert y_windows to pandas Series for efficient mapping if it's not already
                # This is crucial if y_windows contains mixed types or needs flexible lookup
                y_series = pd.Series(y_windows)
                y_windows_mapped = y_series.map(label_mapping_dict).fillna(y_series).values
                
                # Check how many labels were changed
                changed_count = np.sum(y_windows != y_windows_mapped)
                print(f"  Successfully remapped {changed_count} labels out of {len(y_windows)} based on Activity_Mapping_v2.csv.")
                
                # Example: Show some changes if any
                if changed_count > 0:
                    # Find first few differing indices
                    diff_indices = np.where(y_windows != y_windows_mapped)[0]
                    print(f"  Example of remapping (first {min(5, len(diff_indices))} changes):")
                    for i in range(min(5, len(diff_indices))):
                        idx = diff_indices[i]
                        print(f"    Original: '{y_windows[idx]}' -> Mapped: '{y_windows_mapped[idx]}'")
                
                y_windows = y_windows_mapped # Update y_windows with the remapped labels
            else:
                print("  Warning: Activity_Mapping_v2.csv does not contain 'Former_Label' and/or 'New_Label' columns. Skipping remapping.")
        except Exception as e:
            print(f"  Error processing Activity_Mapping_v2.csv: {e}. Skipping remapping.")
    else:
        print(f"  Warning: Activity_Mapping_v2.csv not found at {activity_mapping_path}. Skipping remapping.")
    # --- END LABEL REMAPPING ---

    if not isinstance(X_windows, np.ndarray) or not isinstance(y_windows, np.ndarray) or not isinstance(subject_ids_windows, np.ndarray):
        print("Error: Loaded data is not in the expected NumPy array format.")
        return [None]*8

    # ADDITION: Manually exclude specified subjects from config
    excluded_subjects_manual = config.get('excluded_subjects_manual', ['OutSense-036', 'OutSense-425', 'OutSense-515'])
    if excluded_subjects_manual:
        print(f"Attempting to manually exclude subjects: {excluded_subjects_manual}")
        initial_count = len(subject_ids_windows)
        # Create a boolean mask for subjects NOT in the exclusion list
        exclusion_mask = ~np.isin(subject_ids_windows, excluded_subjects_manual)
        
        X_windows = X_windows[exclusion_mask]
        y_windows = y_windows[exclusion_mask]
        subject_ids_windows = subject_ids_windows[exclusion_mask]
        
        excluded_count = initial_count - len(subject_ids_windows)
        print(f"Manually excluded {excluded_count} data points belonging to specified subjects.")
        
        if X_windows.shape[0] == 0:
            print("Error: All data was excluded after applying 'excluded_subjects_manual'. Please check your config.")
            return [None]*8
        print(f"Data shape after manual exclusion: X={X_windows.shape}, y={y_windows.shape}, subjects={subject_ids_windows.shape[0] if subject_ids_windows.ndim > 0 else 0}")
    # END ADDITION for manual subject exclusion

    # --- START UNCONDITIONAL REMOVAL OF "Other" and "Unknown" --- 
    print("INFO: Unconditionally removing 'Other' and 'Unknown' classes.")
    
    # Remove "Other" class
    other_class_label = "Other"
    if other_class_label in np.unique(y_windows):
        print(f"  Attempting to remove '{other_class_label}' class.")
        initial_count_other = len(y_windows)
        other_mask = y_windows != other_class_label
        X_windows = X_windows[other_mask]
        subject_ids_windows = subject_ids_windows[other_mask]
        y_windows = y_windows[other_mask] # y_windows must be last for correct counts
        removed_count_other = initial_count_other - len(y_windows)
        print(f"  Removed {removed_count_other} windows for '{other_class_label}'. New y_windows count: {len(y_windows)}")
        if len(y_windows) == 0:
            print(f"Error: All data excluded after removing '{other_class_label}' class. Cannot proceed.")
            return [None]*8
    else:
        print(f"  Label '{other_class_label}' not found in current y_windows. No changes made for 'Other' class.")

    # Remove "Unknown" class
    unknown_class_label = "Unknown"
    if unknown_class_label in np.unique(y_windows):
        print(f"  Attempting to remove '{unknown_class_label}' class.")
        initial_count_unknown = len(y_windows)
        unknown_mask = y_windows != unknown_class_label
        X_windows = X_windows[unknown_mask]
        subject_ids_windows = subject_ids_windows[unknown_mask]
        y_windows = y_windows[unknown_mask] # y_windows must be last
        removed_count_unknown = initial_count_unknown - len(y_windows)
        print(f"  Removed {removed_count_unknown} windows for '{unknown_class_label}'. New y_windows count: {len(y_windows)}")
        if len(y_windows) == 0:
            print(f"Error: All data excluded after removing '{unknown_class_label}' class. Cannot proceed.")
            return [None]*8
    else:
        print(f"  Label '{unknown_class_label}' not found in current y_windows. No changes made for 'Unknown' class.")
    # --- END UNCONDITIONAL REMOVAL --- 

    # --- ADDITION: Filter by selected_classes or exclude specific classes (original logic for other exclusions if needed) ---
    selected_classes = config.get('selected_classes', ['Assisted Propulsion', 'Self Propulsion', 'Resting', 'Excercising'])
    
    if selected_classes:
        print(f"INFO: 'selected_classes' is provided. Filtering dataset to include only: {selected_classes}")
        initial_count_y = len(y_windows)
        
        # Create a boolean mask for rows where y_windows is in selected_classes
        class_selection_mask = np.isin(y_windows, selected_classes)
        
        X_windows = X_windows[class_selection_mask]
        y_windows = y_windows[class_selection_mask]
        subject_ids_windows = subject_ids_windows[class_selection_mask]
        
        kept_count_y = len(y_windows)
        print(f"Filtered by 'selected_classes': Kept {kept_count_y} from {initial_count_y} windows.")

        if kept_count_y == 0:
            print("Error: No data remains after filtering by 'selected_classes'. Please check your config.")
            return [None]*8 # Assuming this is the standard error return format
    else:
        print("INFO: 'selected_classes' not provided or empty. Proceeding with exclude_other_class/exclude_unknown_class logic.")
        # Original logic for excluding "Other" and "Unknown" if selected_classes is not used
        # These specific flags for "Other" and "Unknown" are now redundant due to unconditional removal above,
        # but the structure is kept if user wants to use these flags for other labels by changing the string literals.
        exclude_other_class_flag = config.get('exclude_other_class', False)
        # other_class_label is already defined and handled above
        if exclude_other_class_flag:
            print(f"INFO: Config flag 'exclude_other_class' is True, but '{other_class_label}' should have been removed unconditionally. Checking anyway.")
            if other_class_label in np.unique(y_windows):
                print(f"  Warning: '{other_class_label}' found despite unconditional removal. Removing now based on flag.")
                other_mask = y_windows != other_class_label
                X_windows = X_windows[other_mask]
                subject_ids_windows = subject_ids_windows[other_mask]
                y_windows = y_windows[other_mask]
                if len(y_windows) == 0: return [None]*8
            else:
                print(f"  '{other_class_label}' not present (as expected after unconditional removal).")

        exclude_unknown_class_flag = config.get('exclude_unknown_class', False)
        # unknown_class_label is already defined and handled above
        if exclude_unknown_class_flag:
            print(f"INFO: Config flag 'exclude_unknown_class' is True, but '{unknown_class_label}' should have been removed unconditionally. Checking anyway.")
            if unknown_class_label in np.unique(y_windows):
                print(f"  Warning: '{unknown_class_label}' found despite unconditional removal. Removing now based on flag.")
                unknown_mask = y_windows != unknown_class_label
                X_windows = X_windows[unknown_mask]
                subject_ids_windows = subject_ids_windows[unknown_mask]
                y_windows = y_windows[unknown_mask]
                if len(y_windows) == 0: return [None]*8
            else:
                print(f"  '{unknown_class_label}' not present (as expected after unconditional removal).")
    # --- END ADDITION for class filtering ---
    
    # --- ADDITION: Remove classes with fewer than N instances ---
    min_instances_threshold = config.get('min_class_instances', 10) # Default to 10 if not in config
    if min_instances_threshold > 0:
        print(f"INFO: Removing classes with fewer than {min_instances_threshold} instances.")
        unique_labels, counts = np.unique(y_windows, return_counts=True)
        labels_to_remove = unique_labels[counts < min_instances_threshold]

        if len(labels_to_remove) > 0:
            print(f"  Classes to remove (less than {min_instances_threshold} instances): {list(labels_to_remove)}")
            initial_count_y_min_inst = len(y_windows)
            
            # Create a mask to keep only classes that are NOT in labels_to_remove
            min_instances_mask = ~np.isin(y_windows, labels_to_remove)
            
            X_windows = X_windows[min_instances_mask]
            y_windows = y_windows[min_instances_mask]
            subject_ids_windows = subject_ids_windows[min_instances_mask]
            
            kept_count_y_min_inst = len(y_windows)
            removed_count = initial_count_y_min_inst - kept_count_y_min_inst
            print(f"  Removed {removed_count} windows belonging to classes with less than {min_instances_threshold} instances.")

            if kept_count_y_min_inst == 0:
                print(f"Error: No data remains after removing classes with fewer than {min_instances_threshold} instances. Please check your data or config.")
                return [None]*8 # Assuming this is the standard error return format
        else:
            print(f"  No classes found with fewer than {min_instances_threshold} instances. No changes made.")
    else:
        print("INFO: min_class_instances threshold is 0 or not set. Skipping removal of low-instance classes.")
    # --- END ADDITION for removing low-instance classes ---

    # ADDITION: Report on all unique labels in the loaded dataset (NOW AFTER CLASS FILTERING)
    all_original_labels = np.unique(y_windows)
    num_all_original_labels = len(all_original_labels)
    label_display_limit = 10 # For concise printing of label lists
    print(f"Found {num_all_original_labels} unique labels in the original y_windows.")
    if num_all_original_labels > 0:
        display_labels_str = ", ".join(map(str, all_original_labels[:label_display_limit]))
        ellipsis = "..." if num_all_original_labels > label_display_limit else ""
        print(f"  Examples: [{display_labels_str}{ellipsis}]")
    # END ADDITION

    if X_windows.ndim != 3:
        print(f"Error: X_windows (raw data) has incorrect dimensions ({X_windows.ndim}). Expected 3 dimensions (n_windows, window_size, n_features).")
        return [None]*8
    n_features_loaded = X_windows.shape[2]       # Actual number of features from data
    print(f"Loaded {X_windows.shape[0]} windows, with sequence length {X_windows.shape[1]} and {n_features_loaded} initial features.")


    # Ensure X_windows is float32, as expected by later code and PyTorch
    if X_windows.dtype != np.float32:
        X_windows = X_windows.astype(np.float32)

    # 3. Train/Validation/Test Split (remains largely the same, uses loaded subject_ids_windows)
    test_subjects = config.get('test_subjects', [])
    if not test_subjects: 
        print("Error: 'test_subjects' must be defined in config.")
        return [None]*8
    
    # Separate test set first
    test_mask = np.isin(subject_ids_windows, test_subjects)
    train_val_mask = ~test_mask
    
    X_train_val, y_train_val, subject_ids_train_val = X_windows[train_val_mask], y_windows[train_val_mask], subject_ids_windows[train_val_mask]
    X_test, y_test = X_windows[test_mask], y_windows[test_mask]
    
    if len(X_train_val) == 0 or len(X_test) == 0:
        print("Error: Train/validation set or test set is empty after splitting by subject. Check test_subjects and data.")
        return [None]*8

    # Split remaining data into train and validation using GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=50) # test_size=0.1 means 10% of train_val for validation
    
    # Ensure there are enough groups (subjects) for the split
    unique_subjects_train_val = np.unique(subject_ids_train_val)
    if len(unique_subjects_train_val) < 2 and len(X_train_val) > 0 : # gss requires at least 2 groups for a split if data exists
        print("Warning: Not enough unique subjects in the training/validation set for GroupShuffleSplit. Using all for training, validation will be empty.")
        # In this case, validation set will be empty. Handle this or adjust logic.
        # For now, let's proceed, but val_loader might be empty.
        train_idx = np.arange(len(X_train_val))
        val_idx = np.array([], dtype=int) # Empty validation set
    elif len(X_train_val) == 0: # No data to split
        train_idx, val_idx = np.array([], dtype=int), np.array([], dtype=int)
    else:
        train_idx, val_idx = next(gss.split(X_train_val, y_train_val, groups=subject_ids_train_val))

    X_train, y_train = X_train_val[train_idx], y_train_val[train_idx]
    s_train = subject_ids_train_val[train_idx] # ADDED: Keep track of subject IDs for X_train
    X_val, y_val = X_train_val[val_idx], y_train_val[val_idx]
    s_val = subject_ids_train_val[val_idx] # ADDED: Keep track of subject IDs for X_val
    
    print(f"Data split: Train={len(X_train)}, Validation={len(X_val)}, Test={len(X_test)} sequences.")
    if len(X_train) > 0: print(f"Train subject IDs: {np.unique(subject_ids_train_val[train_idx])}")
    if len(X_val) > 0: print(f"Validation subject IDs: {np.unique(subject_ids_train_val[val_idx])}")
    if len(X_test) > 0: print(f"Test subject IDs: {np.unique(subject_ids_windows[test_mask])}")

    # ADDITION: Check label presence in each set
    print("\\n--- Label Presence Check Across Sets ---")
    unique_train_labels = np.array([]) # Initialize, will be updated if train set exists

    if len(X_train) > 0:
        unique_train_labels = np.unique(y_train)
        num_unique_train_labels = len(unique_train_labels)
        print(f"Training set: {num_unique_train_labels} unique labels.")
        if num_unique_train_labels > 0:
            display_labels_str = ", ".join(map(str, unique_train_labels[:label_display_limit]))
            ellipsis = "..." if num_unique_train_labels > label_display_limit else ""
            print(f"  Examples: [{display_labels_str}{ellipsis}]")
        
        missing_in_train = set(all_original_labels) - set(unique_train_labels)
        if missing_in_train:
            missing_list_str = ", ".join(map(str, sorted(list(missing_in_train))[:label_display_limit]))
            ellipsis = "..." if len(missing_in_train) > label_display_limit else ""
            print(f"  WARNING: Training set is missing {len(missing_in_train)} labels from original dataset: [{missing_list_str}{ellipsis}]")
        elif num_all_original_labels > 0 : # Only print if there were original labels to begin with
            print("  All original labels are present in the training set.")
    else:
        print("Training set is empty. No labels to check for training set.")

    if len(X_val) > 0:
        unique_val_labels = np.unique(y_val)
        num_unique_val_labels = len(unique_val_labels)
        print(f"Validation set: {num_unique_val_labels} unique labels.")
        if num_unique_val_labels > 0:
            display_labels_str = ", ".join(map(str, unique_val_labels[:label_display_limit]))
            ellipsis = "..." if num_unique_val_labels > label_display_limit else ""
            print(f"  Examples: [{display_labels_str}{ellipsis}]")

        missing_in_val_vs_original = set(all_original_labels) - set(unique_val_labels)
        if missing_in_val_vs_original:
            missing_list_str = ", ".join(map(str, sorted(list(missing_in_val_vs_original))[:label_display_limit]))
            ellipsis = "..." if len(missing_in_val_vs_original) > label_display_limit else ""
            print(f"  INFO: Validation set is missing {len(missing_in_val_vs_original)} labels from original dataset: [{missing_list_str}{ellipsis}]")
        
        if len(X_train) > 0: # Check against train labels if train set exists
             missing_in_val_vs_train = set(unique_train_labels) - set(unique_val_labels)
             if missing_in_val_vs_train:
                 missing_list_str = ", ".join(map(str, sorted(list(missing_in_val_vs_train))[:label_display_limit]))
                 ellipsis = "..." if len(missing_in_val_vs_train) > label_display_limit else ""
                 print(f"  INFO: Validation set is missing {len(missing_in_val_vs_train)} labels that ARE present in training set: [{missing_list_str}{ellipsis}]")
    else:
        print("Validation set is empty. No labels to check for validation set.")

    if len(X_test) > 0:
        unique_test_labels = np.unique(y_test)
        num_unique_test_labels = len(unique_test_labels)
        print(f"Test set: {num_unique_test_labels} unique labels.")
        if num_unique_test_labels > 0:
            display_labels_str = ", ".join(map(str, unique_test_labels[:label_display_limit]))
            ellipsis = "..." if num_unique_test_labels > label_display_limit else ""
            print(f"  Examples: [{display_labels_str}{ellipsis}]")

        missing_in_test_vs_original = set(all_original_labels) - set(unique_test_labels)
        if missing_in_test_vs_original:
            missing_list_str = ", ".join(map(str, sorted(list(missing_in_test_vs_original))[:label_display_limit]))
            ellipsis = "..." if len(missing_in_test_vs_original) > label_display_limit else ""
            print(f"  INFO: Test set is missing {len(missing_in_test_vs_original)} labels from original dataset: [{missing_list_str}{ellipsis}]")

        if len(X_train) > 0: # Check against train labels if train set exists
            missing_in_test_vs_train = set(unique_train_labels) - set(unique_test_labels)
            if missing_in_test_vs_train:
                 missing_list_str = ", ".join(map(str, sorted(list(missing_in_test_vs_train))[:label_display_limit]))
                 ellipsis = "..." if len(missing_in_test_vs_train) > label_display_limit else ""
                 print(f"  INFO: Test set is missing {len(missing_in_test_vs_train)} labels that ARE present in training set: [{missing_list_str}{ellipsis}]")
    else:
        print("Test set is empty. No labels to check for test set.")
    print("--- End of Label Presence Check ---\\n")
    # END ADDITION

    if len(X_train) == 0:
        print("Error: Training set is empty. Cannot proceed.")
        return [None]*8 # Assuming this is the standard error return format

    # --- START UNDERSAMPLING TRAINING DATA ---
    print(f"Original training set size before undersampling: {len(X_train)}")
    if len(X_train) > 0:
        # Get initial distribution in training set
        unique_classes_in_train, counts_in_train = np.unique(y_train, return_counts=True)
        print(f"Training class distribution before any filtering for undersampling: {dict(zip(unique_classes_in_train, counts_in_train))}")

        # Step 1: Filter out classes from y_train that are below min_instances_threshold for undersampling
        min_instances_for_undersampling = config.get('min_class_instances', 10) 
        
        if min_instances_for_undersampling > 0 and len(unique_classes_in_train) > 0:
            # Identify classes in the current training set that are below the threshold
            labels_to_remove_from_train_mask = counts_in_train < min_instances_for_undersampling
            labels_to_remove_from_train = unique_classes_in_train[labels_to_remove_from_train_mask]

            if len(labels_to_remove_from_train) > 0:
                print(f"  Classes in training set with < {min_instances_for_undersampling} instances will be removed before undersampling: {list(labels_to_remove_from_train)}")
                
                # Create a mask for y_train to keep only classes NOT in labels_to_remove_from_train
                train_keep_mask = ~np.isin(y_train, labels_to_remove_from_train)
                X_train = X_train[train_keep_mask]
                y_train = y_train[train_keep_mask]
                s_train = s_train[train_keep_mask] # MODIFIED: Filter s_train accordingly
                
                if len(y_train) == 0:
                    print("Error: Training set is empty after removing classes with too few instances (before undersampling). Cannot proceed.")
                    return [None]*8 
                
                # Update unique_classes_in_train and counts_in_train after this filtering
                unique_classes_in_train, counts_in_train = np.unique(y_train, return_counts=True)
                print(f"Training class distribution after filtering for >= {min_instances_for_undersampling} instances (before undersampling): {dict(zip(unique_classes_in_train, counts_in_train))}")
            else:
                print(f"  All classes in training set have >= {min_instances_for_undersampling} instances. No pre-filter needed before undersampling.")
        
        # Step 2: Proceed with undersampling based on the (potentially) filtered y_train
        if len(unique_classes_in_train) > 1: # Only undersample if there's more than one class remaining
            min_class_count = np.min(counts_in_train) # Minority count from *remaining* classes in train set
            print(f"Minority class count for undersampling (from classes with >= {min_instances_for_undersampling} instances in train set): {min_class_count}")

            resampled_X_train_list = []
            resampled_y_train_list = []
            resampled_s_train_list = [] # ADDED: List for subject IDs

            for cls_label in unique_classes_in_train: # Iterate over classes that passed the threshold within y_train
                cls_indices = np.where(y_train == cls_label)[0]
                selected_indices = np.random.choice(cls_indices, size=min_class_count, replace=False)
                
                resampled_X_train_list.append(X_train[selected_indices])
                resampled_y_train_list.append(y_train[selected_indices])
                resampled_s_train_list.append(s_train[selected_indices]) # ADDED: Append corresponding subject IDs

            if resampled_X_train_list: 
                X_train = np.concatenate(resampled_X_train_list, axis=0)
                y_train = np.concatenate(resampled_y_train_list, axis=0)
                s_train = np.concatenate(resampled_s_train_list, axis=0) # ADDED: Concatenate subject IDs

                shuffle_indices = np.random.permutation(len(X_train))
                X_train = X_train[shuffle_indices]
                y_train = y_train[shuffle_indices]
                s_train = s_train[shuffle_indices] # ADDED: Shuffle subject IDs consistently

                print(f"Resampled training set size: {len(X_train)}")
                if len(X_train) == 0:
                    print("Error: Training set is empty after undersampling. Cannot proceed.")
                    return [None]*8 
                unique_classes_resampled, counts_resampled = np.unique(y_train, return_counts=True)
                print(f"Resampled training class distribution: {dict(zip(unique_classes_resampled, counts_resampled))}")
            else:
                # This case might occur if unique_classes_in_train was not empty but something went wrong with list appends
                print("Skipping actual concatenation for undersampling as resampled lists are unexpectedly empty.")
        elif len(unique_classes_in_train) == 1:
            print("Skipping undersampling as only one class remains in training data after filtering for min instances.")
        else: # len(unique_classes_in_train) == 0 (i.e., y_train became empty after filtering)
            print("Skipping undersampling as no classes remain in training data after filtering for min instances.")
    else:
        print("Skipping undersampling as training set is empty before undersampling.")
    # --- END UNDERSAMPLING TRAINING DATA ---

    # --- START: Plot selected windows of UNDERSAMPLED TRAINING DATA each class ---
    if X_train.shape[0] > 0 and y_train.shape[0] > 0 and 's_train' in locals() and hasattr(s_train, 'shape') and s_train.shape[0] == X_train.shape[0] and plot_windows_enabled:
        plots_base_dir = os.path.join(project_root, "feature_plots_train_undersampled_by_class")
        os.makedirs(plots_base_dir, exist_ok=True)
        print(f"\nINFO: Plotting selected windows from UNDERSAMPLED TRAINING SET by class to {plots_base_dir}")

        current_unique_labels_for_plotting = np.unique(y_train)
        
        num_features_to_plot = 0
        if X_train.ndim == 3 and X_train.shape[2] > 0:
            num_features_to_plot = X_train.shape[2]
        else:
            print("Warning: X_train is not 3-dimensional or has no features after undersampling. Skipping plotting of window content.")

        if num_features_to_plot > 0: 
            for class_label in current_unique_labels_for_plotting:
                class_dir_name = str(class_label).replace(" ", "_").replace("/", "_").replace("\\", "_")
                class_plot_dir = os.path.join(plots_base_dir, class_dir_name)
                os.makedirs(class_plot_dir, exist_ok=True)
                
                class_indices = np.where(y_train == class_label)[0]
                num_plots_for_class = min(len(class_indices), 50) # Limit plots per class
                print(f"  Plotting {num_plots_for_class} example windows for class '{class_label}' from undersampled training data...")
                
                for i, idx_in_current_X_train in enumerate(class_indices[:num_plots_for_class]): 
                    window_data = X_train[idx_in_current_X_train] 
                    subject_id = s_train[idx_in_current_X_train]

                    plt.figure(figsize=(12, 6))
                    for feature_idx in range(num_features_to_plot):
                        plt.plot(window_data[:, feature_idx], label=f'Feature {feature_idx + 1}')
                    
                    plt.title(f'Undersampled Train - Class: {class_label} - Subject: {subject_id} - Window {i+1}/{len(class_indices)}\n(Idx in Undersampled Train: {idx_in_current_X_train})')
                    plt.xlabel('Time Step in Window')
                    plt.ylabel('Feature Value')
                    if num_features_to_plot <= 10: 
                        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.0))
                    elif num_features_to_plot <= 20: 
                         plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                    
                    plt.grid(True, linestyle='--', alpha=0.7)
                    plt.tight_layout(rect=[0, 0, 0.85, 1] if num_features_to_plot > 10 and num_features_to_plot <=20 else [0,0,1,1] ) 

                    plot_filename = os.path.join(class_plot_dir, f"train_undersampled_window_{i}_subj_{subject_id}.png")
                    try:
                        plt.savefig(plot_filename)
                    except Exception as e:
                        print(f"    Error saving plot {plot_filename}: {e}")
                    plt.close() 
                print(f"  Finished plotting for class '{class_label}'. Plots saved in {class_plot_dir}")
            print(f"--- Finished plotting all selected windows from undersampled training set by class ---\n")
        elif X_train.shape[0] > 0 : 
             print("INFO: Skipping plotting of undersampled training windows as X_train has no features or is not 3D.")
    else:
        print("INFO: Skipping plotting of undersampled training windows as no data remains in X_train or s_train is invalid after undersampling.")
    # --- END: Plot selected windows ---

    # 4. Scale Features
    scaler = StandardScaler()
    # n_train_samples, seq_len, n_feat_scaler = X_train.shape # n_feat_scaler should be n_features_loaded
    # Fit scaler only on the training data's features
    scaler.fit(X_train.reshape(-1, n_features_loaded)) 
    X_train_scaled = scaler.transform(X_train.reshape(-1, n_features_loaded)).reshape(X_train.shape)
    if len(X_val) > 0:
        X_val_scaled = scaler.transform(X_val.reshape(-1, n_features_loaded)).reshape(X_val.shape)
    else:
        X_val_scaled = np.array([]).reshape(0, X_train.shape[1] if X_train.ndim == 3 else 0, n_features_loaded) # Ensure correct shape for empty val
    if len(X_test) > 0:
        X_test_scaled = scaler.transform(X_test.reshape(-1, n_features_loaded)).reshape(X_test.shape)
    else:
        X_test_scaled = np.array([]).reshape(0, X_train.shape[1] if X_train.ndim == 3 else 0, n_features_loaded) # Ensure correct shape for empty test
            
    # 5. Encode Labels
    label_encoder = LabelEncoder()
    # Fit on all possible labels from the y_windows *after* class filtering to ensure consistency
    # This ensures the encoder only knows about the classes that are actually going to be used.
    current_unique_labels = np.unique(y_windows)
    if len(current_unique_labels) == 0:
        print("Error: No labels remaining in y_windows to fit the LabelEncoder. This might be due to over-aggressive filtering.")
        return [None]*8
    label_encoder.fit(current_unique_labels) 
    
    y_train_enc = label_encoder.transform(y_train)
    if len(X_val) > 0:
        y_val_enc = label_encoder.transform(y_val)
    else:
        y_val_enc = np.array([])
    if len(X_test) > 0:
        y_test_enc = label_encoder.transform(y_test)
    else:
        y_test_enc = np.array([])
        
    num_classes = len(label_encoder.classes_)
    print(f"Found {num_classes} classes: {label_encoder.classes_}")

    # 6. Prepare data for PyTorch
    X_train_final = torch.from_numpy(X_train_scaled.transpose(0, 2, 1)).float() # (batch, features, seq_len)
    y_train_final = torch.from_numpy(y_train_enc).long()
    
    if len(X_val) > 0:
        X_val_final = torch.from_numpy(X_val_scaled.transpose(0, 2, 1)).float()
        y_val_final = torch.from_numpy(y_val_enc).long()
    else:            
        # Create empty tensors with correct feature dimension if val set is empty
        X_val_final = torch.empty(0, n_features_loaded, X_train_final.shape[2] if X_train_final.nelement() > 0 else 0).float()
        y_val_final = torch.empty(0).long()

    if len(X_test) > 0:
        X_test_final = torch.from_numpy(X_test_scaled.transpose(0, 2, 1)).float()
        y_test_final = torch.from_numpy(y_test_enc).long()
    else:            
        # Create empty tensors with correct feature dimension if test set is empty
        X_test_final = torch.empty(0, n_features_loaded, X_train_final.shape[2] if X_train_final.nelement() > 0 else 0).float()
        y_test_final = torch.empty(0).long()

    # 7. Create DataLoaders
    train_loader = DataLoader(TensorDataset(X_train_final, y_train_final), batch_size=BATCH_SIZE, shuffle=True)
    
    if len(X_val) > 0:
        val_loader = DataLoader(TensorDataset(X_val_final, y_val_final), batch_size=BATCH_SIZE, shuffle=False)
    else:
        val_loader = None # Or an empty DataLoader if preferred by downstream code
        print("Validation set is empty. val_loader will be None.")

    if len(X_test) > 0:
        test_loader = DataLoader(TensorDataset(X_test_final, y_test_final), batch_size=BATCH_SIZE, shuffle=False)
    else:
        test_loader = None # Should not happen if initial checks are robust
        print("Test set is empty. test_loader will be None.")


    print("Data preparation complete.")
    # Assuming the original return was: train_loader, val_loader, test_loader, scaler, label_encoder, num_classes, n_features_loaded, class_weights_tensor
    return train_loader, val_loader, test_loader, scaler, label_encoder, num_classes, n_features_loaded, None # MODIFIED RETURN


# --- Training and Evaluation Functions (MODIFIED for early stopping) ---
def train_model(model, train_loader, val_loader, optimizer, criterion, device, num_epochs, best_model_save_path, last_model_save_path, use_early_stopping):
    print(f"\\n--- Starting Model Training (Early Stopping: {use_early_stopping}) ---")
    model.to(device)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state_dict = None # Store the state_dict of the best model

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
        
        if use_early_stopping:
            # Early stopping check
            if avg_val_loss < best_val_loss - EARLY_STOPPING_MIN_DELTA:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                # Save the best model state
                best_model_state_dict = model.state_dict()
                print(f"  Validation loss improved. Saving best model state to {best_model_save_path}.")
                torch.save(best_model_state_dict, best_model_save_path)
            else:
                epochs_no_improve += 1
                print(f"  Validation loss did not improve. Counter: {epochs_no_improve}/{EARLY_STOPPING_PATIENCE}")

            if epochs_no_improve == EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered after {EARLY_STOPPING_PATIENCE} epochs with no improvement.")
                break
        else: # If early stopping is not used, save model at each epoch as potentially the best one so far
            if avg_val_loss < best_val_loss: # Still track best_val_loss to save the actual best
                best_val_loss = avg_val_loss
                best_model_state_dict = model.state_dict()
                torch.save(best_model_state_dict, best_model_save_path)
                print(f"  Best model updated at epoch {epoch+1} (Early Stopping OFF). Saved to {best_model_save_path}")

    # Save the last model state
    print(f"Saving last model state to {last_model_save_path}.")
    torch.save(model.state_dict(), last_model_save_path)

    # Load the best model weights back into the model instance if an improvement was found
    if best_model_state_dict:
        print(f"Loading best model weights from {best_model_save_path} for returning.")
        model.load_state_dict(best_model_state_dict)
    else:
        print("Warning: Training finished without improvement or early stopping did not save a new best. Using current (last) model state.")
        # If no improvement was ever seen (best_model_state_dict is None), 
        # the model already has its last state. If early stopping saved one, but then training continued to NUM_EPOCHS
        # without further improvement, best_model_state_dict would hold the best one.
        # This logic ensures that if best_model_state_dict exists, it's loaded.
        # If not, the model (which is the last state) is used as is.
        # The best_model_save_path might still contain a model from a previous run if no improvement happened in this run.

    return model, last_model_save_path # Return the model instance (best weights loaded) and path to last model state


# The evaluate_model function remains the same as the last version (with both "all" and "no unknown" evaluations)
def evaluate_model(model, test_loader, label_encoder, device, filename_suffix=""):
    """
    Evaluates the model on the test set and generates confusion matrices.
    
    This version is optimized for a large number of labels (~170) by:
    - Disabling cell annotations in the heatmap.
    - Displaying only a subset of axis tick labels to prevent overlap.
    - Using a more reasonable figure size.
    """
    print("\n--- Evaluating Model (Standard) ---") # Clarified it's standard evaluation
    model.eval() # Ensure model is in evaluation mode (dropout disabled by default)
    all_preds_encoded, all_labels_encoded = [], []
    
    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences = sequences.to(device)
            outputs = model(sequences) # Single forward pass
            probs = torch.softmax(outputs, dim=-1)
            _, predicted = torch.max(probs, 1)
            
            all_preds_encoded.extend(predicted.cpu().numpy())
            all_labels_encoded.extend(labels.cpu().numpy())

    all_preds_encoded = np.array(all_preds_encoded)
    all_labels_encoded = np.array(all_labels_encoded)
    
    print("\n--- Overall Performance (including 'Unknown' class) ---")
    accuracy_all = accuracy_score(all_labels_encoded, all_preds_encoded)
    f1_all = f1_score(all_labels_encoded, all_preds_encoded, average='weighted')
    print(f"Test Accuracy (All Classes): {accuracy_all:.4f}")
    print(f"Test F1-Score (Weighted, All Classes): {f1_all:.4f}")
    
    cm_all = confusion_matrix(all_labels_encoded, all_preds_encoded, labels=np.arange(len(label_encoder.classes_)))
    
    # --- MODIFIED HEATMAP PLOTTING (All Classes) ---
    plt.figure(figsize=(25, 22))
    
    all_class_names = label_encoder.classes_
    num_all_classes = len(all_class_names)

    if num_all_classes <= 30:
        xticklabels_all = all_class_names
        yticklabels_all = all_class_names
        annot_all = True
    else:
        tick_interval_all = max(1, num_all_classes // 20)
        xticklabels_all = [label if i % tick_interval_all == 0 else "" for i, label in enumerate(all_class_names)]
        yticklabels_all = [label if i % tick_interval_all == 0 else "" for i, label in enumerate(all_class_names)]
        annot_all = False

    sns.heatmap(cm_all, annot=annot_all, cmap='Blues', 
                xticklabels=xticklabels_all, yticklabels=yticklabels_all, fmt='d' if annot_all else '') # Add fmt='d' for integer annotations
    
    plt.title('Confusion Matrix (All Classes)'); plt.ylabel('True Label'); plt.xlabel('Predicted Label')
    cm_all_filename = f"tcn_confusion_matrix_all_classes{filename_suffix}.png"
    plt.savefig(cm_all_filename, dpi=300); plt.close() # Added dpi for better quality
    print(f"Confusion matrix for all classes saved to: {cm_all_filename}")

    # --- MODIFIED TO PLOT NORMALIZED CONFUSION MATRIX (All Classes) ---
    print(f"\n--- Plotting Normalized Confusion Matrix (All Classes) ---")
    cm_all_normalized = cm_all.astype('float') / cm_all.sum(axis=1)[:, np.newaxis]
    # Replace NaN with 0 (happens for rows with no true samples, though ideally all classes in encoder should be in y_true at some point)
    cm_all_normalized = np.nan_to_num(cm_all_normalized, nan=0.0)

    plt.figure(figsize=(25, 22))
    sns.heatmap(cm_all_normalized, annot=annot_all, cmap='Blues', 
                xticklabels=xticklabels_all, yticklabels=yticklabels_all, 
                fmt='.2f' if annot_all else '') # Use .2f for normalized values if annotated
    
    plt.title(f'Normalized Confusion Matrix (All Classes){filename_suffix}'); 
    plt.ylabel('True Label'); 
    plt.xlabel('Predicted Label')
    cm_all_normalized_filename = f"tcn_confusion_matrix_all_classes_normalized{filename_suffix}.png"
    plt.savefig(cm_all_normalized_filename, dpi=300); plt.close()
    print(f"Normalized confusion matrix for all classes saved to: {cm_all_normalized_filename}")


# --- NEW FUNCTION for Uncertainty Evaluation ---
def evaluate_with_uncertainty(model, test_loader, label_encoder, device, filename_suffix=""):
    """
    Evaluates the model using Monte Carlo Dropout to quantify and visualize prediction uncertainty.
    """
    if MC_DROPOUT_SAMPLES <= 0:
        print("\nINFO: MC Dropout is disabled (MC_DROPOUT_SAMPLES=0). Skipping uncertainty evaluation.")
        return
        
    print(f"\n--- Evaluating Model with Uncertainty (MC Dropout with {MC_DROPOUT_SAMPLES} samples) ---")
    
    # 1. Activate dropout layers for inference
    model.enable_dropout()
    
    all_labels_encoded = []
    all_mc_probabilities = [] # Store all raw probability distributions from all passes

    with torch.no_grad():
        for i, (sequences, labels) in enumerate(test_loader):
            sequences = sequences.to(device)
            
            # Store predictions from multiple forward passes for each item in the batch
            mc_outputs_batch = []
            for _ in range(MC_DROPOUT_SAMPLES):
                outputs = model(sequences)
                mc_outputs_batch.append(outputs)
            
            # Stack outputs along a new dimension: (n_samples, batch_size, num_classes)
            mc_outputs_stack = torch.stack(mc_outputs_batch, dim=0)
            
            # Convert logits to probabilities using Softmax
            mc_softmax_stack = torch.softmax(mc_outputs_stack, dim=2) # dim=2 because output is (n_mc_samples, batch, classes)
            
            all_labels_encoded.extend(labels.cpu().numpy())
            all_mc_probabilities.append(mc_softmax_stack.cpu().numpy())

    # Concatenate results from all batches
    all_labels_encoded = np.array(all_labels_encoded)
    # Target shape for all_mc_probabilities after concatenation and transpose:
    # (total_samples, n_mc_samples, num_classes)
    # mc_softmax_stack.cpu().numpy() gives (n_mc_samples, batch_size, num_classes)
    # So, all_mc_probabilities becomes a list of such arrays.
    # np.concatenate(all_mc_probabilities, axis=1) will join along batch_size dimension.
    # Resulting shape: (n_mc_samples, total_samples, num_classes)
    # Then transpose to (total_samples, n_mc_samples, num_classes)
    all_mc_probabilities = np.concatenate(all_mc_probabilities, axis=1).transpose(1, 0, 2)

    # --- Calculate Uncertainty and Final Predictions ---
    
    # Final prediction is the class with the highest mean probability across MC samples
    mean_probs_per_sample = all_mc_probabilities.mean(axis=1) # Shape: (total_samples, num_classes)
    final_preds_encoded = np.argmax(mean_probs_per_sample, axis=1)

    # Uncertainty Metric 1: Predictive Entropy (Total Uncertainty)
    # Higher entropy means the model's averaged prediction is less confident (flatter distribution).
    predictive_entropy = -np.sum(mean_probs_per_sample * np.log(mean_probs_per_sample + 1e-9), axis=1)
    
    # Uncertainty Metric 2: Variance of Probabilities (A measure of model disagreement)
    # Variance across MC samples for each class, then summed up for each data sample.
    variance_per_sample = np.sum(np.var(all_mc_probabilities, axis=1), axis=1)

    # Create a DataFrame for easy analysis and plotting
    uncertainty_df = pd.DataFrame({
        'true_label': label_encoder.inverse_transform(all_labels_encoded),
        'predicted_label': label_encoder.inverse_transform(final_preds_encoded),
        'is_correct': all_labels_encoded == final_preds_encoded,
        'predictive_entropy': predictive_entropy,
        'total_variance': variance_per_sample
    })

    print("\n--- Uncertainty Analysis Summary ---")
    # Ensure all true labels are present in the DataFrame before grouping, 
    # otherwise, classes not present in uncertainty_df['true_label'] might cause errors or be missing.
    # This is generally handled if label_encoder.classes_ are used for ordering in plots.
    # For describe(), it will only show stats for labels present in uncertainty_df['true_label']
    print(uncertainty_df.groupby('true_label')['predictive_entropy'].describe())

    # --- Visualization 1: Boxplot of Uncertainty per True Class ---
    plt.figure(figsize=(18, 10))
    # Order for boxplot should be based on unique labels present in the data to avoid errors
    # if some classes from label_encoder.classes_ are not in uncertainty_df['true_label']
    sorted_true_labels = sorted(uncertainty_df['true_label'].unique())
    sns.boxplot(data=uncertainty_df, x='predictive_entropy', y='true_label', orient='h', 
                order=sorted_true_labels)
    plt.title('Prediction Uncertainty (Predictive Entropy) by True Class', fontsize=16)
    plt.xlabel('Predictive Entropy (Higher = More Uncertain)', fontsize=12)
    plt.ylabel('True Class', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    uncertainty_boxplot_filename = f"tcn_uncertainty_boxplot{filename_suffix}.png"
    plt.savefig(uncertainty_boxplot_filename, dpi=300)
    plt.close()
    print(f"\nUncertainty boxplot saved to: {uncertainty_boxplot_filename}")

    # --- Visualization 2: Density Plot of Uncertainty for Correct vs. Incorrect Predictions ---
    plt.figure(figsize=(12, 7))
    sns.kdeplot(data=uncertainty_df, x='predictive_entropy', hue='is_correct', 
                fill=True, common_norm=False, palette='crest')
    plt.title('Distribution of Uncertainty for Correct vs. Incorrect Predictions', fontsize=16)
    plt.xlabel('Predictive Entropy (Higher = More Uncertain)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    uncertainty_kde_filename = f"tcn_uncertainty_correctness_kde{filename_suffix}.png"
    plt.savefig(uncertainty_kde_filename, dpi=300)
    plt.close()
    print(f"Uncertainty correctness density plot saved to: {uncertainty_kde_filename}")
    
    # Ensure model is set back to eval mode if it was changed
    model.eval()

# --- Main Execution (MODIFIED to use validation loader) ---
if __name__ == "__main__":
    # --- Argument Parser Setup ---
    parser = argparse.ArgumentParser(description='Run TCN for activity classification with optional window plotting.')
    parser.add_argument('--plot-windows', action='store_true', 
                        help='Enable plotting of selected windows after undersampling.')
    args = parser.parse_args()
    plot_windows_enabled = False

    if not os.path.exists(CONFIG_FILE): print(f"Config '{CONFIG_FILE}' not found."); exit()
    with open(CONFIG_FILE, 'r') as f: config = yaml.safe_load(f)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = config.get('project_root_dir', os.path.abspath(os.path.join(script_dir, '..')) if "src" in script_dir.lower() else script_dir)

    # Read early stopping configuration from config.yaml
    # The key in config.yaml is 'use_early_stopping' under 'model_training' or similar section
    # For this script, we assume it might be a top-level or specific TCN training param.
    # Let's assume a structure like: tcn_training_params: use_early_stopping: True
    # Or, if it's a general param: use_early_stopping: True
    # Based on the provided config.yaml, it seems to be under 'Stage 5: Model Training'
    # So, config.get('use_early_stopping', True) might be too general if other models exist.
    # Let's assume for this script, we can add a specific section or use a general one.
    # For now, I will check for a top-level `use_early_stopping` and default to the script's global.
    USE_EARLY_STOPPING_CONFIG = config.get('use_early_stopping', USE_EARLY_STOPPING) # Default to script global if not in config
    # Update script global based on config
    EARLY_STOPPING_PATIENCE = config.get('early_stopping_patience', EARLY_STOPPING_PATIENCE)
    # Note: EARLY_STOPPING_MIN_DELTA is not in the provided config.yaml snippet for Stage 5, so it will use the script default.

    # MODIFIED unpacking - class_weights is now _ (or any other placeholder for None)
    train_dl, val_dl, test_dl, fitted_scaler, fitted_encoder, n_classes, n_features, _ = prepare_data_for_tcn(config, project_root, plot_windows_enabled)
    
    if train_dl is None: 
        print("Data preparation failed. Exiting."); exit()
    if val_dl is None: 
        print("Warning: Validation data loader is not available.") 
    if test_dl is None: 
        print("Warning: Test data loader is not available. Skipping final evaluation on test set.")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\\nUsing device: {device}")
    
    # # Move class weights to the correct device # REMOVED as class weights are no longer used/calculated
    # # if class_weights is not None: # Original code had class_weights.to(device)
    # #     class_weights = class_weights.to(device) 
    # #     print(f"Class weights moved to device: {class_weights.device}")

    # Initialize model for training
    tcn_model_for_training = TCNModel(num_inputs=n_features, num_channels=TCN_NUM_CHANNELS, num_classes=n_classes, 
                         kernel_size=TCN_KERNEL_SIZE, dropout=TCN_DROPOUT)
    
    optimizer = optim.AdamW(tcn_model_for_training.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # MODIFIED criterion - no weights argument
    criterion = nn.CrossEntropyLoss() 

    model_dir_full = os.path.join(project_root, MODELS_OUTPUT_DIR)
    os.makedirs(model_dir_full, exist_ok=True)
    best_model_save_path = os.path.join(model_dir_full, MODEL_FILENAME) # Path for the best model
    last_model_save_path = os.path.join(model_dir_full, LAST_MODEL_FILENAME) # Path for the last model
    
    # Train the model
    # train_model now returns the best model instance and the path to the last model's state dict
    best_tcn_model, saved_last_model_state_path = train_model(
        tcn_model_for_training, 
        train_dl, 
        val_dl, 
        optimizer, 
        criterion, 
        device, 
        NUM_EPOCHS, 
        best_model_save_path, 
        last_model_save_path,
        USE_EARLY_STOPPING_CONFIG # Pass the flag to the training function
    )

    # Evaluate the best model (instance returned by train_model has best weights loaded)
    print("\n--- Evaluating Best Model (from early stopping) ---")
    evaluate_model(best_tcn_model, test_dl, fitted_encoder, device, filename_suffix="_best")

    # --- NEW: Uncertainty Evaluation of Best Model ---
    evaluate_with_uncertainty(
        best_tcn_model, 
        test_dl, 
        fitted_encoder, 
        device, 
        filename_suffix="_best"
    )

    # Evaluate the last model
    if os.path.exists(saved_last_model_state_path):
        print(f"\n--- Evaluating Last Model (from epoch {NUM_EPOCHS}) ---")
        # Create a new model instance for the last model to ensure clean state loading
        last_tcn_model = TCNModel(num_inputs=n_features, num_channels=TCN_NUM_CHANNELS, num_classes=n_classes, 
                                 kernel_size=TCN_KERNEL_SIZE, dropout=TCN_DROPOUT)
        last_tcn_model.load_state_dict(torch.load(saved_last_model_state_path, map_location=device)) # Load onto the correct device
        last_tcn_model.to(device) # Ensure model is on the correct device
        evaluate_model(last_tcn_model, test_dl, fitted_encoder, device, filename_suffix="_last")
    else:
        print(f"Could not find saved last model state at {saved_last_model_state_path}. Skipping evaluation of last model.")

    # Save processors
    scaler_save_path = os.path.join(model_dir_full, SCALER_FILENAME)
    encoder_save_path = os.path.join(model_dir_full, ENCODER_FILENAME)
    with open(scaler_save_path, 'wb') as f: pickle.dump(fitted_scaler, f)
    with open(encoder_save_path, 'wb') as f: pickle.dump(fitted_encoder, f)

    print(f"\n--- Training and Evaluation Complete ---")
    print(f"Best TCN model saved to: {best_model_save_path}")
    print(f"Last TCN model state saved to: {saved_last_model_state_path}")
    print(f"Scaler saved to: {scaler_save_path}")
    print(f"Label Encoder saved to: {encoder_save_path}")