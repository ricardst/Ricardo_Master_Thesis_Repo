# filepath: /scai_data3/scratch/stirnimann_r/src/tcn_activity_classification_CVtuning_boosted.py
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import weight_norm
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
# from sklearn.utils.class_weight import compute_class_weight # Not used in the HPO version
import yaml
import joblib # For saving scaler/encoder
import copy
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress the specific FutureWarning from torch.nn.utils.weight_norm
warnings.filterwarnings("ignore", message="`torch.nn.utils.weight_norm` is deprecated in favor of `torch.nn.utils.parametrizations.weight_norm`.", category=FutureWarning, module="torch.nn.utils.weight_norm")

# --- Configuration (User to modify these) ---
CONFIG_FILE = "config.yaml" # Assumed to be at project root

# -- TCN Model Hyperparameters (These will be tuned by Optuna) --
# TCN_NUM_CHANNELS = [32, 64, 64, 128] 
# TCN_KERNEL_SIZE = 3
# TCN_DROPOUT = 0.35

# -- Training Hyperparameters (Learning rate, weight decay, batch size will be tuned) --
# LEARNING_RATE = 0.0005
# WEIGHT_DECAY = 0.0005 # L2 regularization
# BATCH_SIZE = 64
NUM_EPOCHS = 150 # Max epochs for final training, early stopping will determine actual
# -- Early Stopping Parameters --
EARLY_STOPPING_PATIENCE = 20 
EARLY_STOPPING_MIN_DELTA = 0.0001

# -- Dynamic Loss Weighting Parameters --
DYNAMIC_LOSS_WEIGHTING = True  # Enable/disable dynamic loss weighting
ALPHA_LOSS_WEIGHTING = 2.0  # Alpha parameter for dynamic loss weighting (1-3 recommended)

# -- Noise-Robust Loss Parameters --
USE_GCE_LOSS = True  # Use Generalized Cross Entropy instead of standard Cross Entropy
GCE_Q_PARAMETER = 0.7  # GCE q parameter (0 < q < 1), lower values more robust to noise 

# -- Monte Carlo Dropout Configuration --
MC_DROPOUT_SAMPLES = 100  # Number of forward passes for Monte Carlo Dropout. Set to 0 to disable.

# -- Output Paths --
MODELS_OUTPUT_DIR = "models" # Relative to project_root
# Generate model filenames based on loss function configuration
loss_suffix = "_gce" if USE_GCE_LOSS else "_ce"
dynamic_suffix = "_dynamic" if DYNAMIC_LOSS_WEIGHTING else ""
MODEL_FILENAME = f"tcn_classifier_best_model_tuned{loss_suffix}{dynamic_suffix}.pth"
LAST_MODEL_FILENAME = f"tcn_classifier_last_model_tuned{loss_suffix}{dynamic_suffix}.pth"
SCALER_FILENAME = f"scaler_for_tcn_tuned{loss_suffix}{dynamic_suffix}.pkl"
ENCODER_FILENAME = f"label_encoder_for_tcn_tuned{loss_suffix}{dynamic_suffix}.pkl"

# -- Hyperparameter Tuning Configuration --
NUM_OPTUNA_TRIALS = 32 # Number of HPO trials to run
CV_N_SPLITS = 5 # Number of folds for cross-validation during HPO
HPO_PATIENCE = 10 # Early stopping patience within HPO folds
HPO_MAX_EPOCHS = 20 # Max epochs within HPO folds

# --- TCN Model Definition ---
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
        if self.downsample is not None: 
            self.downsample.weight.data.normal_(0, 0.01)

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
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
            
        self.tcn = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], num_classes); self.init_weights()

    def init_weights(self): self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x): y = self.tcn(x); out = self.linear(y[:, :, -1]); return out

    def enable_dropout(self):
        """ Function to enable the dropout layers during inference. """
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

# --- Data Preparation Function (MODIFIED for HPO) ---
def prepare_data_for_tcn_hpo(config, project_root):
    print("--- Starting Data Preparation for TCN (Harness for HPO) ---")
    
    intermediate_dir = os.path.join(project_root, config.get('intermediate_feature_dir', 'features'))
    x_windows_path = os.path.join(intermediate_dir, "X_windows_raw.npy")
    y_windows_path = os.path.join(intermediate_dir, "y_windows.npy")     
    subject_ids_path = os.path.join(intermediate_dir, "subject_ids_windows.npy")

    required_files = [x_windows_path, y_windows_path, subject_ids_path]
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"ERROR: Required data file not found: {file_path}")
            return [None]*7 # Expected number of return values

    try:
        # Load .npy files using np.load()
        X_windows = np.load(x_windows_path, allow_pickle=True)
        y_windows = np.load(y_windows_path, allow_pickle=True)
        subject_ids_windows = np.load(subject_ids_path, allow_pickle=True)
    except Exception as e:
        print(f"Error loading data from .npy files: {e}")
        return [None]*7 # Assuming 7 items are returned by this function on success.

    # --- START LABEL REMAPPING ---
    print("INFO: Attempting to remap activity labels using Activity_Mapping_v2.csv")
    activity_mapping_path = os.path.join(project_root, "Activity_Mapping_v2.csv") # project_root is a param
    if os.path.exists(activity_mapping_path):
        try:
            mapping_df = pd.read_csv(activity_mapping_path) # pd is pandas
            # Ensure column names match your CSV file exactly.
            if 'Former_Label' in mapping_df.columns and 'New_Label' in mapping_df.columns:
                label_mapping_dict = pd.Series(mapping_df.New_Label.values, index=mapping_df.Former_Label).to_dict()
                
                y_series = pd.Series(y_windows) # y_windows is loaded above
                y_windows_mapped = y_series.map(label_mapping_dict).fillna(y_series).values # np for np.sum, np.where
                
                changed_count = np.sum(y_windows != y_windows_mapped) # np
                print(f"  Successfully remapped {changed_count} labels out of {len(y_windows)} based on Activity_Mapping_v2.csv.")
                
                if changed_count > 0:
                    diff_indices = np.where(y_windows != y_windows_mapped)[0] # np
                    print(f"  Example of remapping (first {min(5, len(diff_indices))} changes):")
                    for i in range(min(5, len(diff_indices))):
                        idx = diff_indices[i]
                        print(f"    Original: '{y_windows[idx]}' -> Mapped: '{y_windows_mapped[idx]}'")
                
                y_windows = y_windows_mapped # Update y_windows
            else:
                print("  Warning: Activity_Mapping_v2.csv does not contain 'Former_Label' and/or 'New_Label' columns. Skipping remapping.")
        except Exception as e:
            print(f"  Error processing Activity_Mapping_v2.csv: {e}. Skipping remapping.")
    else:
        print(f"  Warning: Activity_Mapping_v2.csv not found at {activity_mapping_path}. Skipping remapping.")
    # --- END LABEL REMAPPING ---

    if not isinstance(X_windows, np.ndarray) or not isinstance(y_windows, np.ndarray) or not isinstance(subject_ids_windows, np.ndarray):
        print("Error: Loaded data is not in the expected NumPy array format.")
        return [None]*7

    excluded_subjects_manual = config.get('excluded_subjects_manual', ['OutSense-036', 'OutSense-425', 'OutSense-515']) # Default to empty list
    if excluded_subjects_manual:
        print(f"Attempting to manually exclude subjects: {excluded_subjects_manual}")
        initial_count = len(subject_ids_windows)
        exclusion_mask = ~np.isin(subject_ids_windows, excluded_subjects_manual)
        X_windows = X_windows[exclusion_mask]
        y_windows = y_windows[exclusion_mask]
        subject_ids_windows = subject_ids_windows[exclusion_mask]
        excluded_count = initial_count - len(subject_ids_windows)
        print(f"Manually excluded {excluded_count} data points belonging to specified subjects.")
        if X_windows.shape[0] == 0:
            print("ERROR: All data excluded after manual subject exclusion. Cannot proceed.")
            return [None]*7
        print(f"Data shape after manual exclusion: X={X_windows.shape}, y={y_windows.shape}, subjects={subject_ids_windows.shape[0] if subject_ids_windows.ndim > 0 else 0}")

    print("INFO: Unconditionally removing 'Other' and 'Unknown' classes.")
    other_class_label = "Other"
    if other_class_label in np.unique(y_windows):
        initial_count_other = len(y_windows)
        other_mask = y_windows != other_class_label
        X_windows = X_windows[other_mask]; subject_ids_windows = subject_ids_windows[other_mask]; y_windows = y_windows[other_mask]
        print(f"  Removed {initial_count_other - len(y_windows)} windows for '{other_class_label}'.")
        if len(y_windows) == 0: print("ERROR: All data removed after 'Other' class exclusion."); return [None]*7
    
    unknown_class_label = "Unknown"
    if unknown_class_label in np.unique(y_windows):
        initial_count_unknown = len(y_windows)
        unknown_mask = y_windows != unknown_class_label
        X_windows = X_windows[unknown_mask]; subject_ids_windows = subject_ids_windows[unknown_mask]; y_windows = y_windows[unknown_mask]
        print(f"  Removed {initial_count_unknown - len(y_windows)} windows for '{unknown_class_label}'.")
        if len(y_windows) == 0: print("ERROR: All data removed after 'Unknown' class exclusion."); return [None]*7

    selected_classes = config.get('selected_classes', ['Propulsion', 'Resting', 'Transfer', 'Exercising', 'Conversation']) # Default to None to use all remaining
    if selected_classes:
        print(f"INFO: Filtering dataset to include only: {selected_classes}")
        initial_count_y = len(y_windows)
        class_selection_mask = np.isin(y_windows, selected_classes)
        X_windows = X_windows[class_selection_mask]; y_windows = y_windows[class_selection_mask]; subject_ids_windows = subject_ids_windows[class_selection_mask]
        print(f"Filtered by 'selected_classes': Kept {len(y_windows)} from {initial_count_y} windows.")
        if len(y_windows) == 0: print("ERROR: All data removed after 'selected_classes' filtering."); return [None]*7
    
    min_instances_threshold = config.get('min_class_instances', 10)
    if min_instances_threshold > 0 and len(y_windows) > 0:
        print(f"INFO: Removing classes with fewer than {min_instances_threshold} instances.")
        unique_labels, counts = np.unique(y_windows, return_counts=True)
        labels_to_keep = unique_labels[counts >= min_instances_threshold]
        if len(labels_to_keep) < len(unique_labels):
            instance_mask = np.isin(y_windows, labels_to_keep)
            X_windows = X_windows[instance_mask]; y_windows = y_windows[instance_mask]; subject_ids_windows = subject_ids_windows[instance_mask]
            print(f"  Removed {len(unique_labels) - len(labels_to_keep)} classes. Kept {len(labels_to_keep)} classes.")
            if len(y_windows) == 0: print("ERROR: All data removed after min_class_instances filtering."); return [None]*7
        else:
            print("  All classes meet the minimum instance threshold.")

    if X_windows.ndim != 3:
        print(f"ERROR: X_windows has incorrect dimensions {X_windows.ndim}, expected 3 (samples, sequence_length, features).")
        return [None]*7
    n_features_loaded = X_windows.shape[2]
    print(f"Loaded {X_windows.shape[0]} windows, seq_len {X_windows.shape[1]}, {n_features_loaded} features after filtering.")

    if X_windows.dtype != np.float32:
        print(f"INFO: Converting X_windows from {X_windows.dtype} to np.float32.")
        X_windows = X_windows.astype(np.float32)

    test_subjects = config.get('test_subjects', [])
    if not test_subjects: 
        print("ERROR: 'test_subjects' not defined in config. Cannot proceed.")
        return [None]*7
    
    test_mask = np.isin(subject_ids_windows, test_subjects)
    train_val_mask = ~test_mask
    
    X_train_val_all = X_windows[train_val_mask]
    y_train_val_all = y_windows[train_val_mask]
    subject_ids_train_val_all = subject_ids_windows[train_val_mask]
    
    X_test = X_windows[test_mask]
    y_test = y_windows[test_mask]
    
    if len(X_train_val_all) == 0:
        print("ERROR: No data remains for training/validation after test set separation.")
    if len(X_test) == 0:
        print("WARNING: No data for the test set based on provided test_subjects.")

    print(f"Data split for HPO: Train/Val Pool={len(X_train_val_all)}, Test={len(X_test)} sequences.")

    # Determine all unique labels present in the data that will be used by an encoder
    # This should happen *after* all filtering and *before* any encoding.
    # Consider labels from both train_val_all and test to ensure encoder consistency.
    combined_y_for_labels = []
    if len(y_train_val_all) > 0: combined_y_for_labels.append(y_train_val_all)
    if len(y_test) > 0: combined_y_for_labels.append(y_test)

    if not combined_y_for_labels:
        print("ERROR: No labels found in either training/validation pool or test set after filtering.")
        return [None]*7
        
    all_potential_labels = np.unique(np.concatenate(combined_y_for_labels))
    
    if len(all_potential_labels) == 0:
        print("ERROR: No unique labels found after combining train/val and test sets.")
        return [None]*7
        
    print(f"Found {len(all_potential_labels)} unique labels across combined data for encoder fitting: {all_potential_labels[:10]}...")

    print("Data preparation for HPO harness complete.")
    return X_train_val_all, y_train_val_all, subject_ids_train_val_all, X_test, y_test, n_features_loaded, all_potential_labels

# --- Dynamic Loss Weighting Function ---
def compute_dynamic_weighted_loss(criterion, outputs, targets, alpha=2.0, use_weighting=True):
    """
    Compute dynamically weighted loss based on per-sample confidence.
    
    Args:
        criterion: Loss function (e.g., CrossEntropyLoss or GeneralizedCrossEntropyLoss)
        outputs: Model predictions (logits)
        targets: Ground truth labels
        alpha: Weighting parameter (1-3 recommended)
        use_weighting: Whether to apply dynamic weighting
    
    Returns:
        Weighted loss tensor
    """
    if not use_weighting:
        return criterion(outputs, targets)
    
    # Compute per-sample loss (no reduction)
    if isinstance(criterion, GeneralizedCrossEntropyLoss):
        # GCE loss already supports reduction='none'
        original_reduction = criterion.reduction
        criterion.reduction = 'none'
        per_sample_loss = criterion(outputs, targets)
        criterion.reduction = original_reduction  # Restore original reduction
    else:
        # Standard CrossEntropyLoss
        criterion_no_reduction = nn.CrossEntropyLoss(reduction='none')
        per_sample_loss = criterion_no_reduction(outputs, targets)
    
    # Compute dynamic weights: smaller weight for high loss samples
    weights = torch.exp(-alpha * per_sample_loss)
    
    # Apply weights and compute mean
    weighted_loss = (weights * per_sample_loss).mean()
    
    return weighted_loss

# --- Generalized Cross Entropy Loss ---
class GeneralizedCrossEntropyLoss(nn.Module):
    """
    Generalized Cross Entropy Loss for noise-robust training.
    
    GCE(p, y) = (1 - p_y^q) / q
    where 0 < q < 1. Lower q discounts high-confidence mistakes (often noisy).
    
    Args:
        q: Parameter controlling noise robustness (0 < q < 1)
        reduction: Specifies the reduction to apply to the output
    """
    def __init__(self, q=0.7, reduction='mean'):
        super(GeneralizedCrossEntropyLoss, self).__init__()
        self.q = q
        self.reduction = reduction
        
    def forward(self, outputs, targets):
        # Get softmax probabilities
        probs = torch.softmax(outputs, dim=1)
        
        # Get the probabilities for the true classes
        true_class_probs = torch.gather(probs, 1, targets.unsqueeze(1)).squeeze(1)
        
        # Clamp probabilities to avoid numerical issues
        true_class_probs = torch.clamp(true_class_probs, min=1e-7, max=1.0)
        
        # Compute GCE loss: (1 - p_y^q) / q
        loss = (1.0 - torch.pow(true_class_probs, self.q)) / self.q
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")

# --- Helper Function to Create Loss Function ---
def create_loss_function(use_gce=False, gce_q=0.7):
    """
    Create the appropriate loss function based on configuration.
    
    Args:
        use_gce: Whether to use Generalized Cross Entropy
        gce_q: Q parameter for GCE loss
    
    Returns:
        Loss function instance
    """
    if use_gce:
        return GeneralizedCrossEntropyLoss(q=gce_q, reduction='mean')
    else:
        return nn.CrossEntropyLoss()

# --- Optuna Objective Function ---
def objective(trial, X_tv_data, y_tv_data, groups_tv_data, n_features, potential_labels_for_encoder_fit, device, min_class_instances_config): # ADDED min_class_instances_config
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    tcn_kernel_size = trial.suggest_categorical("tcn_kernel_size", [2, 3, 5])
    
    # Tune number of TCN layers (2-6 layers)
    num_tcn_layers = trial.suggest_int("num_tcn_layers", 2, 6)
    
    # Define channels for maximum possible layers (6), then use only the first num_tcn_layers
    # This ensures consistent parameter names across trials
    max_layers = 6
    all_channels = []
    for layer_idx in range(max_layers):
        if layer_idx == 0:
            # First layer - smaller channels
            channels = trial.suggest_categorical(f"num_channels_l{layer_idx+1}", [16, 32, 64])
        elif layer_idx < max_layers - 1:
            # Middle layers - medium channels
            channels = trial.suggest_categorical(f"num_channels_l{layer_idx+1}", [32, 64, 128])
        else:
            # Last layer - larger channels
            channels = trial.suggest_categorical(f"num_channels_l{layer_idx+1}", [64, 128, 256])
        all_channels.append(channels)
    
    # Use only the first num_tcn_layers channels
    tcn_num_channels = all_channels[:num_tcn_layers]
    
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    
    # Dynamic loss weighting alpha parameter
    alpha_loss_weight = trial.suggest_float("alpha_loss_weight", 1.0, 3.0) if DYNAMIC_LOSS_WEIGHTING else ALPHA_LOSS_WEIGHTING
    
    # GCE loss q parameter
    gce_q_param = trial.suggest_float("gce_q_param", 0.3, 0.9) if USE_GCE_LOSS else GCE_Q_PARAMETER
    
    # Learning rate scheduler hyperparameters
    use_scheduler = trial.suggest_categorical("use_scheduler", [True, False])
    scheduler_type = trial.suggest_categorical("scheduler_type", ["ReduceLROnPlateau", "CosineAnnealingLR"]) if use_scheduler else None
    
    # Scheduler-specific parameters
    scheduler_params = {}
    if use_scheduler:
        if scheduler_type == "ReduceLROnPlateau":
            scheduler_params["factor"] = trial.suggest_float("scheduler_factor", 0.1, 0.8)
            scheduler_params["patience"] = trial.suggest_int("scheduler_patience", 3, 8)
            scheduler_params["min_lr"] = trial.suggest_float("scheduler_min_lr", 1e-7, 1e-5, log=True)
        elif scheduler_type == "CosineAnnealingLR":
            scheduler_params["T_max"] = trial.suggest_int("scheduler_T_max", 10, HPO_MAX_EPOCHS)
            scheduler_params["eta_min"] = trial.suggest_float("scheduler_eta_min", 1e-7, 1e-5, log=True)

    gkf = GroupKFold(n_splits=CV_N_SPLITS)
    fold_f1_scores = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_tv_data, y_tv_data, groups_tv_data)):
        print(f"--- HPO Trial {trial.number}, Fold {fold+1}/{CV_N_SPLITS} ---")
        X_train_fold, X_val_fold = X_tv_data[train_idx], X_tv_data[val_idx]
        y_train_fold, y_val_fold = y_tv_data[train_idx], y_tv_data[val_idx]
        groups_train_fold = groups_tv_data[train_idx]

        # --- START UNDERSAMPLING FOR THIS FOLD'S TRAINING DATA ---
        if len(X_train_fold) > 0:
            unique_classes_fold, counts_fold = np.unique(y_train_fold, return_counts=True)
            print(f"    Fold {fold+1} train class distribution before undersampling: {dict(zip(unique_classes_fold, counts_fold))}")

            # Filter out classes below min_class_instances_config for undersampling
            if min_class_instances_config > 0 and len(unique_classes_fold) > 0:
                labels_to_remove_mask = counts_fold < min_class_instances_config
                labels_to_remove = unique_classes_fold[labels_to_remove_mask]
                if len(labels_to_remove) > 0:
                    print(f"      Removing classes from fold train set (for undersampling) with < {min_class_instances_config} instances: {list(labels_to_remove)}")
                    train_keep_mask = ~np.isin(y_train_fold, labels_to_remove)
                    X_train_fold = X_train_fold[train_keep_mask]
                    y_train_fold = y_train_fold[train_keep_mask]
                    groups_train_fold = groups_train_fold[train_keep_mask]
                    if len(y_train_fold) == 0:
                        print("      Warning: Fold training set empty after removing low-instance classes. Skipping fold.")
                        continue # Skip to next fold
                    unique_classes_fold, counts_fold = np.unique(y_train_fold, return_counts=True)
                    print(f"      Fold {fold+1} train class distribution after filtering for >= {min_class_instances_config} instances: {dict(zip(unique_classes_fold, counts_fold))}")

            # Proceed with undersampling if multiple classes remain
            if len(unique_classes_fold) > 1:
                min_class_count_fold = np.min(counts_fold)
                print(f"    Minority class count for undersampling in fold {fold+1}: {min_class_count_fold}")
                
                resampled_X_fold_list, resampled_y_fold_list, resampled_groups_fold_list = [], [], []
                for cls_label in unique_classes_fold:
                    cls_indices = np.where(y_train_fold == cls_label)[0]
                    selected_indices = np.random.choice(cls_indices, size=min_class_count_fold, replace=False)
                    resampled_X_fold_list.append(X_train_fold[selected_indices])
                    resampled_y_fold_list.append(y_train_fold[selected_indices])
                    resampled_groups_fold_list.append(groups_train_fold[selected_indices])
                
                if resampled_X_fold_list:
                    X_train_fold = np.concatenate(resampled_X_fold_list, axis=0)
                    y_train_fold = np.concatenate(resampled_y_fold_list, axis=0)
                    groups_train_fold = np.concatenate(resampled_groups_fold_list, axis=0)

                    shuffle_indices = np.random.permutation(len(X_train_fold))
                    X_train_fold = X_train_fold[shuffle_indices]
                    y_train_fold = y_train_fold[shuffle_indices]
                    groups_train_fold = groups_train_fold[shuffle_indices]
                    print(f"    Resampled fold {fold+1} training set size: {len(X_train_fold)}")
                    if len(X_train_fold) == 0:
                        print("      Warning: Fold training set empty after undersampling. Skipping fold.")
                        continue # Skip to next fold
                    unique_classes_resampled_f, counts_resampled_f = np.unique(y_train_fold, return_counts=True)
                    print(f"    Resampled fold {fold+1} train class distribution: {dict(zip(unique_classes_resampled_f, counts_resampled_f))}")
            elif len(unique_classes_fold) == 1:
                print(f"    Skipping undersampling for fold {fold+1} as only one class remains after filtering.")
            else:
                print(f"    Skipping undersampling for fold {fold+1} as no classes remain after filtering.")
                if len(X_train_fold) == 0: # Double check if it became empty
                    print("      Warning: Fold training set is empty before model training. Skipping fold.")
                    continue # Skip to next fold
        else: # X_train_fold was initially empty
            print(f"    Skipping undersampling for fold {fold+1} as training set is initially empty.")
            continue # Skip to next fold
        # --- END UNDERSAMPLING FOR THIS FOLD'S TRAINING DATA ---

        # Scaler and LabelEncoder for the fold
        scaler_fold = StandardScaler()
        X_train_fold_scaled = scaler_fold.fit_transform(X_train_fold.reshape(-1, n_features)).reshape(X_train_fold.shape)
        X_val_fold_scaled = scaler_fold.transform(X_val_fold.reshape(-1, n_features)).reshape(X_val_fold.shape)

        label_encoder_fold = LabelEncoder()
        label_encoder_fold.fit(potential_labels_for_encoder_fit)
        y_train_fold_enc = label_encoder_fold.transform(y_train_fold)
        y_val_fold_enc = label_encoder_fold.transform(y_val_fold)
        num_classes_fold = len(label_encoder_fold.classes_)

        X_train_tensor = torch.from_numpy(X_train_fold_scaled.transpose(0, 2, 1)).float()
        y_train_tensor = torch.from_numpy(y_train_fold_enc).long()
        train_loader_fold = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)

        X_val_tensor = torch.from_numpy(X_val_fold_scaled.transpose(0, 2, 1)).float()
        y_val_tensor = torch.from_numpy(y_val_fold_enc).long()
        val_loader_fold = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=batch_size, shuffle=False)

        model_fold = TCNModel(num_inputs=n_features, num_channels=tcn_num_channels, 
                              num_classes=num_classes_fold, kernel_size=tcn_kernel_size, 
                              dropout=dropout_rate).to(device)
        optimizer_fold = optim.AdamW(model_fold.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Choose loss function based on configuration
        if USE_GCE_LOSS:
            criterion_fold = GeneralizedCrossEntropyLoss(q=gce_q_param, reduction='mean')
        else:
            criterion_fold = nn.CrossEntropyLoss()

        # Initialize scheduler if enabled
        scheduler_fold = None
        if use_scheduler:
            if scheduler_type == "ReduceLROnPlateau":
                scheduler_fold = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer_fold, 
                    mode='min', 
                    factor=scheduler_params["factor"],
                    patience=scheduler_params["patience"],
                    min_lr=scheduler_params["min_lr"],
                    verbose=False
                )
            elif scheduler_type == "CosineAnnealingLR":
                scheduler_fold = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer_fold,
                    T_max=scheduler_params["T_max"],
                    eta_min=scheduler_params["eta_min"]
                )

        best_val_loss_fold = float('inf')
        epochs_no_improve_fold = 0
        
        for epoch in range(HPO_MAX_EPOCHS):
            model_fold.train()
            train_loss_epoch = 0
            for batch_X, batch_y in train_loader_fold:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer_fold.zero_grad()
                outputs = model_fold(batch_X)
                # Use dynamic loss weighting with tuned alpha
                loss = compute_dynamic_weighted_loss(criterion_fold, outputs, batch_y, 
                                                   alpha=alpha_loss_weight, 
                                                   use_weighting=DYNAMIC_LOSS_WEIGHTING)
                loss.backward(); optimizer_fold.step(); train_loss_epoch += loss.item()
            
            model_fold.eval(); val_loss_epoch = 0
            with torch.no_grad():
                for batch_X_val, batch_y_val in val_loader_fold:
                    batch_X_val, batch_y_val = batch_X_val.to(device), batch_y_val.to(device)
                    outputs_val = model_fold(batch_X_val)
                    # Use dynamic loss weighting for validation with tuned alpha
                    loss_val = compute_dynamic_weighted_loss(criterion_fold, outputs_val, batch_y_val,
                                                           alpha=alpha_loss_weight,
                                                           use_weighting=DYNAMIC_LOSS_WEIGHTING)
                    val_loss_epoch += loss_val.item()
            avg_val_loss = val_loss_epoch / len(val_loader_fold) if len(val_loader_fold) > 0 else float('inf')
            
            # Step scheduler
            if scheduler_fold is not None:
                if scheduler_type == "ReduceLROnPlateau":
                    scheduler_fold.step(avg_val_loss)
                elif scheduler_type == "CosineAnnealingLR":
                    scheduler_fold.step()
            
            # print(f"T{trial.number} F{fold+1} E{epoch+1} - TrL: {train_loss_epoch/len(train_loader_fold):.4f}, VaL: {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss_fold - EARLY_STOPPING_MIN_DELTA: # Using global min_delta
                best_val_loss_fold = avg_val_loss; epochs_no_improve_fold = 0
            else:
                epochs_no_improve_fold += 1
            if epochs_no_improve_fold >= HPO_PATIENCE: print(f"Early stopping E{epoch+1} T{trial.number} F{fold+1}."); break
        
        model_fold.eval(); all_preds_fold, all_labels_fold = [], []
        with torch.no_grad():
            for batch_X, batch_y in val_loader_fold:
                outputs = model_fold(batch_X.to(device)); preds = torch.argmax(outputs, dim=1)
                all_preds_fold.extend(preds.cpu().numpy()); all_labels_fold.extend(batch_y.cpu().numpy())
        
        f1_val_fold = f1_score(all_labels_fold, all_preds_fold, average='weighted', zero_division=0) if len(all_labels_fold) > 0 else 0.0
        fold_f1_scores.append(f1_val_fold)
        print(f"T{trial.number} F{fold+1} Val F1: {f1_val_fold:.4f}")
        trial.report(f1_val_fold, fold)
        if trial.should_prune(): print(f"T{trial.number} pruned F{fold+1}."); raise optuna.TrialPruned()

    avg_f1 = np.mean(fold_f1_scores) if fold_f1_scores else 0.0
    print(f"T{trial.number} Avg Val F1: {avg_f1:.4f}")
    return avg_f1

# --- Training Function (from original, adapted for early stopping) ---
def train_model(model, train_loader, val_loader, optimizer, criterion, device, num_epochs, best_model_save_path, last_model_save_path, scheduler=None, alpha_loss_weight=None):
    print("\\n--- Starting Model Training with Early Stopping ---")
    print(f"Loss function: {type(criterion).__name__}")
    if isinstance(criterion, GeneralizedCrossEntropyLoss):
        print(f"GCE q parameter: {criterion.q:.3f}")
    if DYNAMIC_LOSS_WEIGHTING:
        alpha_to_use = alpha_loss_weight if alpha_loss_weight is not None else ALPHA_LOSS_WEIGHTING
        print(f"Dynamic loss weighting enabled with alpha: {alpha_to_use:.3f}")
    
    model.to(device)
    
    # Use provided alpha or default
    alpha_to_use = alpha_loss_weight if alpha_loss_weight is not None else ALPHA_LOSS_WEIGHTING
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state_dict = None 

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            # Use dynamic loss weighting with final tuned alpha
            loss = compute_dynamic_weighted_loss(criterion, outputs, batch_y,
                                               alpha=alpha_to_use,
                                               use_weighting=DYNAMIC_LOSS_WEIGHTING)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        
        # Validation
        model.eval()
        epoch_val_loss = 0
        if val_loader: # Only validate if val_loader is provided
            with torch.no_grad():
                for batch_X_val, batch_y_val in val_loader:
                    batch_X_val, batch_y_val = batch_X_val.to(device), batch_y_val.to(device)
                    outputs_val = model(batch_X_val)
                    # Use dynamic loss weighting for validation with final tuned alpha
                    loss_val = compute_dynamic_weighted_loss(criterion, outputs_val, batch_y_val,
                                                           alpha=alpha_to_use,
                                                           use_weighting=DYNAMIC_LOSS_WEIGHTING)
                    epoch_val_loss += loss_val.item()
            avg_val_loss = epoch_val_loss / len(val_loader)
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            # Step scheduler if provided
            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(avg_val_loss)
                else:  # CosineAnnealingLR or other schedulers
                    scheduler.step()

            if avg_val_loss < best_val_loss - EARLY_STOPPING_MIN_DELTA:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                best_model_state_dict = copy.deepcopy(model.state_dict())
                torch.save(best_model_state_dict, best_model_save_path)
                print(f"Validation loss improved. Saved best model to {best_model_save_path}")
            else:
                epochs_no_improve += 1
        else: # No validation loader, just train and log
             print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} (No validation)")
             # Step scheduler even without validation for schedulers that don't need validation loss
             if scheduler is not None and not isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                 scheduler.step()


        if val_loader and epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}.")
            break
    
    print(f"Saving last model state to {last_model_save_path}.")
    torch.save(model.state_dict(), last_model_save_path)

    if best_model_state_dict:
        print("Loading best model weights for returning.")
        model.load_state_dict(best_model_state_dict)
    else: # Should only happen if no val_loader or no improvement ever
        print("No best model state recorded (or no validation), returning model in its last state.")
        if os.path.exists(best_model_save_path) and val_loader: # If best was saved but loop ended due to num_epochs
             model.load_state_dict(torch.load(best_model_save_path)) # Ensure best is loaded
        # If no val_loader, best_model_save_path might not be meaningful for "best"
        # In this case, the model is already in its last state.

    return model, last_model_save_path


# --- Evaluation Functions (from original) ---
def evaluate_model(model, test_loader, label_encoder, device, filename_suffix=""):
    print(f"\\n--- Evaluating Model (Standard) {filename_suffix} ---")
    model.eval() 
    all_preds_encoded, all_labels_encoded = [], []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            preds = torch.argmax(outputs, dim=1)
            all_preds_encoded.extend(preds.cpu().numpy())
            all_labels_encoded.extend(batch_y.cpu().numpy())

    if not all_labels_encoded:
        print("No labels in test set for evaluation. Skipping.")
        return

    all_preds_encoded = np.array(all_preds_encoded)
    all_labels_encoded = np.array(all_labels_encoded)
    
    print(f"\\n--- Overall Performance {filename_suffix} ---")
    accuracy_all = accuracy_score(all_labels_encoded, all_preds_encoded)
    f1_all = f1_score(all_labels_encoded, all_preds_encoded, average='weighted', zero_division=0)
    print(f"Test Accuracy: {accuracy_all:.4f}")
    print(f"Test F1-Score (Weighted): {f1_all:.4f}")
    
    # Step 1: Detailed Per-Class Performance Metrics
    print("\n--- Per-Class Performance ---")
    # Use the label_encoder to get the actual class names
    class_names = label_encoder.classes_ 
    report = classification_report(all_labels_encoded, all_preds_encoded, target_names=class_names, zero_division=0)
    print(report)

    # Save the classification report to a file
    with open(f"tcn_classification_report{filename_suffix}.txt", "w") as f:
        f.write(report)
    print(f"Classification report saved to tcn_classification_report{filename_suffix}.txt")
    
    cm_all = confusion_matrix(all_labels_encoded, all_preds_encoded, labels=np.arange(len(label_encoder.classes_)))
    
    plt.figure(figsize=(25, 22))
    all_class_names = label_encoder.classes_
    num_all_classes = len(all_class_names)
    annot_all = num_all_classes <= 30
    tick_step = 1 if num_all_classes <= 30 else max(1, num_all_classes // 30)
    xticklabels_all = all_class_names[::tick_step] if num_all_classes > 30 else all_class_names
    yticklabels_all = all_class_names[::tick_step] if num_all_classes > 30 else all_class_names
    
    sns.heatmap(cm_all, annot=annot_all, cmap='Blues', 
                xticklabels=xticklabels_all, yticklabels=yticklabels_all, fmt='d' if annot_all else '')
    plt.title(f'Confusion Matrix{filename_suffix}'); plt.ylabel('True Label'); plt.xlabel('Predicted Label')
    cm_all_filename = f"tcn_confusion_matrix{filename_suffix}.png"
    plt.savefig(cm_all_filename, dpi=300, bbox_inches='tight'); plt.close()
    print(f"Confusion matrix saved to: {cm_all_filename}")

    cm_all_normalized = cm_all.astype('float') / cm_all.sum(axis=1)[:, np.newaxis]
    cm_all_normalized = np.nan_to_num(cm_all_normalized, nan=0.0)
    plt.figure(figsize=(25, 22))
    sns.heatmap(cm_all_normalized, annot=annot_all, cmap='Blues', 
                xticklabels=xticklabels_all, yticklabels=yticklabels_all, fmt='.2f' if annot_all else '')
    plt.title(f'Normalized Confusion Matrix{filename_suffix}'); plt.ylabel('True Label'); plt.xlabel('Predicted Label')
    cm_all_normalized_filename = f"tcn_normalized_confusion_matrix{filename_suffix}.png"
    plt.savefig(cm_all_normalized_filename, dpi=300, bbox_inches='tight'); plt.close()
    print(f"Normalized confusion matrix saved to: {cm_all_normalized_filename}")

def evaluate_with_uncertainty(model, test_loader, label_encoder, device, filename_suffix=""):
    if MC_DROPOUT_SAMPLES <= 0:
        print("MC_DROPOUT_SAMPLES is 0 or less. Skipping uncertainty evaluation.")
        return
        
    print(f"\\n--- Evaluating Model with Uncertainty (MC Dropout {MC_DROPOUT_SAMPLES} samples) {filename_suffix} ---")
    model.enable_dropout()
    all_labels_encoded = []
    all_mc_softmax_probs_list = [] # List to store softmax probs from each batch

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            batch_mc_softmax_probs = [] # (n_mc_samples, batch_size, num_classes)
            for _ in range(MC_DROPOUT_SAMPLES):
                outputs = model(batch_X)
                softmax_probs = torch.softmax(outputs, dim=1)
                batch_mc_softmax_probs.append(softmax_probs.cpu().numpy())
            
            all_mc_softmax_probs_list.append(np.array(batch_mc_softmax_probs))
            all_labels_encoded.extend(batch_y.cpu().numpy()) # batch_y is already on CPU from DataLoader or moved

    if not all_labels_encoded:
        print("No labels in test set for uncertainty evaluation. Skipping.")
        model.eval() # Set back to eval mode
        return

    # Concatenate along the batch dimension (axis=1 because shape is (n_mc_samples, batch_size, num_classes))
    all_mc_probabilities = np.concatenate(all_mc_softmax_probs_list, axis=1) 
    # Transpose to (total_samples, n_mc_samples, num_classes)
    all_mc_probabilities = all_mc_probabilities.transpose(1, 0, 2)
    all_labels_encoded = np.array(all_labels_encoded)

    mean_probs_per_sample = all_mc_probabilities.mean(axis=1) 
    final_preds_encoded = np.argmax(mean_probs_per_sample, axis=1)
    predictive_entropy = -np.sum(mean_probs_per_sample * np.log(mean_probs_per_sample + 1e-9), axis=1)
    
    uncertainty_df = pd.DataFrame({
        'true_label': label_encoder.inverse_transform(all_labels_encoded),
        'predicted_label': label_encoder.inverse_transform(final_preds_encoded),
        'is_correct': all_labels_encoded == final_preds_encoded,
        'predictive_entropy': predictive_entropy
    })

    print("\\n--- Uncertainty Analysis Summary ---")
    if not uncertainty_df.empty:
        print(uncertainty_df.groupby('true_label')['predictive_entropy'].describe())
        
        # Step 2: Granular Uncertainty Analysis
        print("\n--- Granular Uncertainty Analysis ---")
        # 1. Average uncertainty per true class
        avg_uncertainty_by_class = uncertainty_df.groupby('true_label')['predictive_entropy'].mean().sort_values(ascending=False)
        print("\nAverage Predictive Entropy by True Class (Higher is more uncertain):")
        print(avg_uncertainty_by_class)

        # 2. Average uncertainty for correct vs. incorrect predictions per class
        uncertainty_by_correctness = uncertainty_df.groupby(['true_label', 'is_correct'])['predictive_entropy'].mean().unstack()
        print("\nAverage Predictive Entropy (Correct vs. Incorrect):")
        print(uncertainty_by_correctness)

        # Save these results to CSV for easier analysis
        avg_uncertainty_by_class.to_csv(f"tcn_avg_uncertainty_by_class{filename_suffix}.csv")
        uncertainty_by_correctness.to_csv(f"tcn_uncertainty_by_correctness{filename_suffix}.csv")
        print(f"Uncertainty analysis data saved to CSV files with suffix: {filename_suffix}")
    else:
        print("Uncertainty DataFrame is empty.")

    if not uncertainty_df.empty and 'predictive_entropy' in uncertainty_df.columns and 'true_label' in uncertainty_df.columns:
        # 3. Enhanced Visualization - Split boxplot showing correct vs incorrect
        plt.figure(figsize=(18, 12))
        sns.boxplot(data=uncertainty_df, x='predictive_entropy', y='true_label', hue='is_correct', 
                   orient='h', order=sorted(uncertainty_df['true_label'].unique()))
        plt.title(f'Uncertainty by Class (Correct vs. Incorrect Predictions){filename_suffix}', fontsize=16)
        plt.xlabel('Predictive Entropy', fontsize=12)
        plt.ylabel('True Class', fontsize=12)
        plt.legend(title='Is Correct')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        uncertainty_split_boxplot_filename = f"tcn_uncertainty_split_boxplot{filename_suffix}.png"
        plt.savefig(uncertainty_split_boxplot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Split uncertainty boxplot saved to: {uncertainty_split_boxplot_filename}")
        
        # Original uncertainty boxplot by class
        plt.figure(figsize=(18, 10))
        sorted_true_labels = sorted(uncertainty_df['true_label'].unique())
        sns.boxplot(data=uncertainty_df, x='predictive_entropy', y='true_label', orient='h', order=sorted_true_labels)
        plt.title(f'Prediction Uncertainty (Predictive Entropy) by True Class{filename_suffix}', fontsize=16)
        plt.xlabel('Predictive Entropy', fontsize=12); plt.ylabel('True Class', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6); plt.tight_layout()
        uncertainty_boxplot_filename = f"tcn_uncertainty_boxplot{filename_suffix}.png"
        plt.savefig(uncertainty_boxplot_filename, dpi=300); plt.close()
        print(f"Uncertainty boxplot saved to: {uncertainty_boxplot_filename}")

        plt.figure(figsize=(12, 7))
        sns.kdeplot(data=uncertainty_df, x='predictive_entropy', hue='is_correct', fill=True, common_norm=False, palette='crest')
        plt.title(f'Distribution of Uncertainty (Correct vs. Incorrect){filename_suffix}', fontsize=16)
        plt.xlabel('Predictive Entropy', fontsize=12); plt.ylabel('Density', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6); plt.tight_layout()
        uncertainty_kde_filename = f"tcn_uncertainty_kde{filename_suffix}.png"
        plt.savefig(uncertainty_kde_filename, dpi=300); plt.close()
        print(f"Uncertainty KDE plot saved to: {uncertainty_kde_filename}")
    else:
        print("Skipping uncertainty plot generation due to empty DataFrame or missing columns.")
    
    model.eval() # Ensure model is back in eval mode

# --- Main Execution (MODIFIED for HPO) ---
if __name__ == "__main__":
    # Determine project root and load config
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Basic check for project root, assuming src is a subdir of project root
    inferred_project_root = os.path.abspath(os.path.join(script_dir, '..')) if "src" in script_dir.lower() else script_dir
    
    config_path = os.path.join(inferred_project_root, CONFIG_FILE)
    if not os.path.exists(config_path):
        print(f"ERROR: Config file {config_path} not found.")
        exit()
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    project_root = config.get('project_root_dir', inferred_project_root)
    print(f"Using project root: {project_root}")

    # --- ADDED: Get min_class_instances from config for undersampling ---
    min_class_instances_for_undersampling = config.get('min_class_instances', 10) # Default to 10 if not in config
    print(f"INFO: Using min_class_instances_for_undersampling = {min_class_instances_for_undersampling} for HPO folds.")
    # --- END ADDED ---

    prepared_data = prepare_data_for_tcn_hpo(config, project_root)
    if any(item is None for item in prepared_data):
        print("ERROR: Data preparation failed. Exiting.")
        exit()
    
    X_tv_all, y_tv_all, groups_tv_all, X_test_final_raw, y_test_final_raw, n_features_loaded, potential_labels = prepared_data

    if len(X_tv_all) == 0:
        print("ERROR: No data available for HPO and training. Exiting.")
        exit()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\\nUsing device: {device}")

    print("\\n--- Starting Hyperparameter Optimization with Optuna ---")
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: objective(trial, X_tv_all, y_tv_all, groups_tv_all, n_features_loaded, potential_labels, device, min_class_instances_for_undersampling), # MODIFIED: Pass min_class_instances_for_undersampling
                   n_trials=NUM_OPTUNA_TRIALS, n_jobs=16) # Changed n_jobs to 1 for sequential processing if issues arise with parallelism

    best_params = study.best_params
    print(f"\\n--- Optuna HPO Complete ---")
    print(f"Best trial: {study.best_trial.number}, F1: {study.best_value:.4f}")
    print(f"Best hyperparameters:")
    print(f"  Learning rate: {best_params['lr']:.6f}")
    print(f"  Weight decay: {best_params['weight_decay']:.6f}")
    print(f"  Dropout rate: {best_params['dropout_rate']:.3f}")
    print(f"  TCN kernel size: {best_params['tcn_kernel_size']}")
    print(f"  Number of TCN layers: {best_params['num_tcn_layers']}")
    tcn_channels_str = [str(best_params[f'num_channels_l{i+1}']) for i in range(best_params['num_tcn_layers'])]
    print(f"  TCN channels: [{', '.join(tcn_channels_str)}]")
    print(f"  Batch size: {best_params['batch_size']}")
    if DYNAMIC_LOSS_WEIGHTING:
        print(f"  Alpha loss weighting: {best_params.get('alpha_loss_weight', ALPHA_LOSS_WEIGHTING):.3f}")
    if USE_GCE_LOSS:
        print(f"  GCE q parameter: {best_params.get('gce_q_param', GCE_Q_PARAMETER):.3f}")
    print(f"  Use scheduler: {best_params['use_scheduler']}")
    if best_params['use_scheduler']:
        print(f"  Scheduler type: {best_params['scheduler_type']}")
        if best_params['scheduler_type'] == 'ReduceLROnPlateau':
            print(f"    Factor: {best_params['scheduler_factor']:.3f}")
            print(f"    Patience: {best_params['scheduler_patience']}")
            print(f"    Min LR: {best_params['scheduler_min_lr']:.2e}")
        elif best_params['scheduler_type'] == 'CosineAnnealingLR':
            print(f"    T_max: {best_params['scheduler_T_max']}")
            print(f"    Eta_min: {best_params['scheduler_eta_min']:.2e}")

    print("\\n--- Training Final Model with Best Hyperparameters ---")
    final_lr = best_params['lr']
    final_weight_decay = best_params['weight_decay']
    final_dropout = best_params['dropout_rate']
    final_tcn_kernel_size = best_params['tcn_kernel_size']
    final_batch_size = best_params['batch_size']
    final_alpha_loss_weight = best_params.get('alpha_loss_weight', ALPHA_LOSS_WEIGHTING) if DYNAMIC_LOSS_WEIGHTING else ALPHA_LOSS_WEIGHTING
    final_gce_q_param = best_params.get('gce_q_param', GCE_Q_PARAMETER) if USE_GCE_LOSS else GCE_Q_PARAMETER
    
    # Build final TCN channels list based on tuned number of layers
    final_num_tcn_layers = best_params['num_tcn_layers']
    final_tcn_channels = []
    for layer_idx in range(final_num_tcn_layers):
        final_tcn_channels.append(best_params[f'num_channels_l{layer_idx+1}'])
    
    # Extract scheduler parameters
    final_use_scheduler = best_params['use_scheduler']
    final_scheduler_type = best_params.get('scheduler_type') if final_use_scheduler else None
    final_scheduler_params = {}
    if final_use_scheduler and final_scheduler_type:
        if final_scheduler_type == "ReduceLROnPlateau":
            final_scheduler_params = {
                "factor": best_params["scheduler_factor"],
                "patience": best_params["scheduler_patience"],
                "min_lr": best_params["scheduler_min_lr"]
            }
        elif final_scheduler_type == "CosineAnnealingLR":
            final_scheduler_params = {
                "T_max": best_params["scheduler_T_max"],
                "eta_min": best_params["scheduler_eta_min"]
            }
    
    gss_final_split = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    
    if len(np.unique(groups_tv_all)) < 2 and len(X_tv_all) > 0 :        
        print("Warning: Only one group for GroupShuffleSplit. Using random split for final validation.")
        from sklearn.model_selection import train_test_split
        final_train_idx, final_val_idx = train_test_split(np.arange(len(X_tv_all)), test_size=0.15, random_state=42, stratify=y_tv_all if len(np.unique(y_tv_all)) > 1 else None)
    elif len(X_tv_all) == 0:        
        print("ERROR: X_tv_all is empty before final split. Cannot proceed."); exit()
    else:
        final_train_idx, final_val_idx = next(gss_final_split.split(X_tv_all, y_tv_all, groups_tv_all))

    X_final_train_raw, X_final_val_raw = X_tv_all[final_train_idx], X_tv_all[final_val_idx]
    y_final_train_raw, y_final_val_raw = y_tv_all[final_train_idx], y_tv_all[final_val_idx]
    print(f"Final data split: Train={len(X_final_train_raw)}, Val={len(X_final_val_raw)}, Test={len(X_test_final_raw)}")

    if len(X_final_train_raw) == 0: print("ERROR: Final training set is empty."); exit()

    final_scaler = StandardScaler()
    X_final_train_scaled = final_scaler.fit_transform(X_final_train_raw.reshape(-1, n_features_loaded)).reshape(X_final_train_raw.shape)
    X_final_val_scaled = final_scaler.transform(X_final_val_raw.reshape(-1, n_features_loaded)).reshape(X_final_val_raw.shape) if len(X_final_val_raw) > 0 else np.array([])
    X_test_final_scaled = final_scaler.transform(X_test_final_raw.reshape(-1, n_features_loaded)).reshape(X_test_final_raw.shape) if len(X_test_final_raw) > 0 else np.array([])

    final_label_encoder = LabelEncoder()
    final_label_encoder.fit(potential_labels)
    y_final_train_enc = final_label_encoder.transform(y_final_train_raw)
    num_classes_final = len(final_label_encoder.classes_)
    print(f"Final model classes ({num_classes_final}): {final_label_encoder.classes_[:10]}...")
    y_final_val_enc = final_label_encoder.transform(y_final_val_raw) if len(X_final_val_raw) > 0 else np.array([])
    y_test_final_enc = final_label_encoder.transform(y_test_final_raw) if len(X_test_final_raw) > 0 else np.array([])

    final_train_loader = DataLoader(TensorDataset(torch.from_numpy(X_final_train_scaled.transpose(0,2,1)).float(), torch.from_numpy(y_final_train_enc).long()), batch_size=final_batch_size, shuffle=True)
    final_val_loader = DataLoader(TensorDataset(torch.from_numpy(X_final_val_scaled.transpose(0,2,1)).float(), torch.from_numpy(y_final_val_enc).long()), batch_size=final_batch_size, shuffle=False) if len(X_final_val_raw) > 0 else None
    final_test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test_final_scaled.transpose(0,2,1)).float(), torch.from_numpy(y_test_final_enc).long()), batch_size=final_batch_size, shuffle=False) if len(X_test_final_raw) > 0 else None

    final_tcn_model = TCNModel(num_inputs=n_features_loaded, num_channels=final_tcn_channels, num_classes=num_classes_final, kernel_size=final_tcn_kernel_size, dropout=final_dropout)
    final_optimizer = optim.AdamW(final_tcn_model.parameters(), lr=final_lr, weight_decay=final_weight_decay)
    
    # Choose final loss function based on configuration
    if USE_GCE_LOSS:
        final_criterion = GeneralizedCrossEntropyLoss(q=final_gce_q_param, reduction='mean')
        print(f"Using Generalized Cross Entropy Loss with q={final_gce_q_param:.3f}")
    else:
        final_criterion = nn.CrossEntropyLoss()
        print("Using standard Cross Entropy Loss")

    # Initialize scheduler if enabled
    final_scheduler = None
    if final_use_scheduler and final_scheduler_type:
        if final_scheduler_type == "ReduceLROnPlateau":
            final_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                final_optimizer, 
                mode='min', 
                factor=final_scheduler_params["factor"],
                patience=final_scheduler_params["patience"],
                min_lr=final_scheduler_params["min_lr"],
                verbose=True
            )
        elif final_scheduler_type == "CosineAnnealingLR":
            final_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                final_optimizer,
                T_max=final_scheduler_params["T_max"],
                eta_min=final_scheduler_params["eta_min"]
            )

    model_dir_full = os.path.join(project_root, MODELS_OUTPUT_DIR)
    os.makedirs(model_dir_full, exist_ok=True)
    best_model_save_path = os.path.join(model_dir_full, MODEL_FILENAME)
    last_model_save_path = os.path.join(model_dir_full, LAST_MODEL_FILENAME)
    
    trained_best_model, saved_last_model_path = train_model(final_tcn_model, final_train_loader, final_val_loader, final_optimizer, final_criterion, device, NUM_EPOCHS, best_model_save_path, last_model_save_path, scheduler=final_scheduler, alpha_loss_weight=final_alpha_loss_weight)

    if final_test_loader and trained_best_model:
        print("\\n--- Evaluating Best Tuned Model on Test Set ---")
        # Create evaluation suffix based on configuration
        eval_suffix = "_best_tuned"
        if USE_GCE_LOSS:
            eval_suffix += "_gce"
        else:
            eval_suffix += "_ce"
        if DYNAMIC_LOSS_WEIGHTING:
            eval_suffix += "_dynamic"
            
        evaluate_model(trained_best_model, final_test_loader, final_label_encoder, device, filename_suffix=eval_suffix)
        evaluate_with_uncertainty(trained_best_model, final_test_loader, final_label_encoder, device, filename_suffix=eval_suffix)
    else:
        print("Skipping final evaluation on test set (no test data or model training failed).")

    scaler_save_path = os.path.join(model_dir_full, SCALER_FILENAME)
    encoder_save_path = os.path.join(model_dir_full, ENCODER_FILENAME)
    with open(scaler_save_path, 'wb') as f: joblib.dump(final_scaler, f)
    with open(encoder_save_path, 'wb') as f: joblib.dump(final_label_encoder, f)

    print(f"\\n--- Tuned Pipeline Complete ---")
    if os.path.exists(best_model_save_path): print(f"Best tuned model: {best_model_save_path}")
    if os.path.exists(saved_last_model_path): print(f"Last tuned model state: {saved_last_model_path}")
    print(f"Tuned Scaler: {scaler_save_path}"); print(f"Tuned Label Encoder: {encoder_save_path}")

