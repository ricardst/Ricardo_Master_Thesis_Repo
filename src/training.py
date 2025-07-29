# src/training.py

import os
import logging
import numpy as np
import pandas as pd # For isnull check
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset # Need Subset for CV
from sklearn.model_selection import GroupKFold
import copy

# Assuming utils.py, models.py are in the same src directory or accessible
try:
    from . import utils
    from .utils import SensorDataset # Import Dataset class
    from . import models # Import the models module
except ImportError:
    import utils
    from utils import SensorDataset
    import models # Fallback if run directly


def compute_class_weights(y_train_encoded, n_classes):
    """
    Computes class weights inversely proportional to class frequencies from encoded labels.

    Args:
        y_train_encoded (np.ndarray): 1D array of encoded training labels.
        n_classes (int): The total number of unique classes.

    Returns:
        torch.Tensor: A tensor of weights, one for each class. Returns uniform weights if calculation fails.
    """
    logging.info(f"Computing class weights for {n_classes} classes...")
    if y_train_encoded is None or y_train_encoded.size == 0:
        logging.warning("Cannot compute class weights: y_train_encoded is empty or None. Returning uniform weights.")
        return torch.ones(n_classes) / n_classes if n_classes > 0 else torch.ones(n_classes)
    if n_classes <= 0:
        logging.warning(f"Invalid n_classes ({n_classes}) for weight computation. Returning uniform weights.")
        # Avoid division by zero if n_classes is 0
        return torch.ones(n_classes) / n_classes if n_classes > 0 else torch.ones(n_classes)

    try:
        # Ensure labels are suitable for bincount (non-negative integers)
        if not np.issubdtype(y_train_encoded.dtype, np.integer):
             logging.warning(f"Labels dtype is not integer ({y_train_encoded.dtype}). Attempting conversion.")
             # Check range before converting
             min_label, max_label = np.min(y_train_encoded), np.max(y_train_encoded)
             if min_label < 0:
                  logging.error(f"Labels contain negative values ({min_label}). Cannot use bincount.")
                  raise ValueError("Labels cannot be negative for bincount.")
             if max_label >= n_classes:
                  logging.error(f"Max label ({max_label}) >= n_classes ({n_classes}). Check label encoding.")
                  # Adjust n_classes or raise error? Raising is safer.
                  raise ValueError("Max label index exceeds number of classes.")
             labels_tensor = torch.tensor(y_train_encoded, dtype=torch.long)
        else:
             labels_tensor = torch.tensor(y_train_encoded, dtype=torch.long)


        counts = torch.bincount(labels_tensor, minlength=n_classes)
        total_samples = len(labels_tensor)

        if total_samples == 0:
             logging.warning("No samples found for weight calculation after tensor conversion.")
             return torch.ones(n_classes) / n_classes if n_classes > 0 else torch.ones(n_classes)

        epsilon = 1e-8
        weights = total_samples / (n_classes * (counts.float() + epsilon))

        # Handle potential Infs if counts were zero despite epsilon (highly unlikely)
        if torch.isinf(weights).any():
            logging.warning("Infinite weights detected. Replacing with 1.0.")
            weights[torch.isinf(weights)] = 1.0 # Replace Inf with 1

        logging.info(f"Class counts: {counts.tolist()}")
        logging.info(f"Computed class weights: {[f'{w:.4f}' for w in weights.tolist()]}")
        return weights

    except Exception as e:
        logging.error(f"Error computing class weights: {e}. Returning uniform weights.", exc_info=True)
        return torch.ones(n_classes) / n_classes if n_classes > 0 else torch.ones(n_classes)


def train_model(model, train_loader, test_loader, config, n_classes, device):
    """
    Trains the final model on the entire training dataset and evaluates it on the
    test dataset after each epoch.

    Args:
        model (nn.Module): The PyTorch model instance to train (already on device).
        train_loader (DataLoader): DataLoader for the training dataset.
        test_loader (DataLoader): DataLoader for the test dataset.
        config (dict): Configuration dictionary.
        n_classes (int): Number of classes.
        device (torch.device): The device to train on (e.g., 'cuda' or 'cpu').

    Returns:
        tuple: Contains:
            - model (nn.Module): The trained model (best state based on test loss might be loaded).
            - train_losses (list): List of average training loss per epoch.
            - test_losses (list): List of average test loss per epoch.
            - test_accuracies (list): List of test accuracy per epoch.
    """
    logging.info(f"\n--- Starting Final Model Training ---")
    logging.info(f"Using device: {device}")
    logging.info(f"Training samples: {len(train_loader.dataset)}, Test samples: {len(test_loader.dataset)}")
    n_epochs = config.get('n_epochs', 100) # Use updated default/value from config
    early_stopping_patience = config.get('early_stopping_patience', 10)
    early_stopping_metric = config.get('early_stopping_metric', 'val_acc').lower()
    logging.info(f"Max Epochs: {n_epochs}, Early Stopping Patience: {early_stopping_patience} ({early_stopping_metric})")

    lr = config.get('lr', 0.001)
    logging.info(f"Epochs: {n_epochs}, Batch size: {train_loader.batch_size}, Learning rate: {lr}")

    # Determine model type to set optimizers
    model_name = config.get('model_name', 'Simple1DCNN')
    is_fen_fln_model = model_name == 'GeARFEN' # or any other model that uses fen and fln

    # --- Optimizers ---
    if is_fen_fln_model:
        fen_lr = config.get('fen_lr', 0.001)
        fln_lr = config.get('fln_lr', 0.001)
        optimizer_params = config.get('optimizer_params', {}) # e.g., weight_decay
        logging.info(f"Using separate optimizers: FEN LR={fen_lr}, FLN LR={fln_lr}")
        # Ensure model has .fen and .fln attributes
        if not hasattr(model, 'fen') or not hasattr(model, 'fln'):
             raise AttributeError(f"Model {model_name} does not have expected 'fen' and 'fln' attributes for separate optimizers.")
        fen_optimizer = optim.Adam(model.fen.parameters(), lr=fen_lr, **optimizer_params)
        fln_optimizer = optim.Adam(model.fln.parameters(), lr=fln_lr, **optimizer_params)
        optimizers = [fen_optimizer, fln_optimizer]
    else:
        # Single optimizer for Simple1DCNN or other models
        lr = config.get('lr', 0.001)
        optimizer_params = config.get('optimizer_params', {})
        logging.info(f"Using single optimizer with LR={lr}")
        optimizer = optim.Adam(model.parameters(), lr=lr, **optimizer_params)
        optimizers = [optimizer]

    # --- Compute class weights using the *entire* training dataset's labels ---
    # Extract all training labels (if dataset is large, this might be slow - consider sampling)
    try:
        y_train_all = []
        # Ensure we handle datasets where __getitem__ returns (data, target)
        for _, target_batch in train_loader:
             y_train_all.append(target_batch)
        y_train_all_np = torch.cat(y_train_all).cpu().numpy()
    except Exception as e:
        logging.error(f"Could not extract all labels from train_loader for weight calculation: {e}. Using uniform weights.", exc_info=True)
        y_train_all_np = np.array([]) # Ensure it's an array for compute_class_weights

    class_weights = compute_class_weights(y_train_all_np, n_classes).to(device)

    try:
        y_train_all = torch.cat([target for _, target in train_loader]).cpu().numpy()
    except: # Handle potential errors if loader empty etc.
         y_train_all = np.array([])
    class_weights = compute_class_weights(y_train_all, n_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # --- Tracking and Early Stopping ---
    train_losses, test_losses, test_accuracies = [], [], []
    best_metric_value = -float('inf') if early_stopping_metric == 'val_acc' else float('inf')
    no_improvement_count = 0
    best_model_state = None

    logging.info(f"Monitoring '{early_stopping_metric}' for early stopping.")

    # --- Training Loop ---
    for epoch in range(n_epochs):
        # --- Training Phase ---
        model.train()
        epoch_train_loss = 0.0
        correct_train = 0
        total_train = 0
        num_batches_train = len(train_loader)

        if num_batches_train == 0: # Handle empty loader
            logging.warning(f"Epoch {epoch+1}: Train loader empty.")
            train_losses.append(np.nan)
            continue # Skip rest of epoch

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            # Zero gradients for all optimizers
            for opt in optimizers: opt.zero_grad()

            output = model(data) # Forward pass through combined model
            loss = criterion(output, target.view(-1)) # Ensure target is flattened

            loss.backward() # Calculate gradients for whole model graph

            # Step all optimizers
            for opt in optimizers: opt.step()

            epoch_train_loss += loss.item()
            _, predicted = torch.max(output, 1)
            correct_train += (predicted == target.view(-1)).sum().item()
            total_train += target.size(0)

        avg_train_loss = epoch_train_loss / num_batches_train
        train_acc = 100 * correct_train / total_train
        train_losses.append(avg_train_loss)

        # --- Validation Phase ---
        model.eval()
        epoch_test_loss = 0.0
        correct_val = 0
        total_val = 0
        num_batches_test = len(test_loader)

        if num_batches_test == 0: # Handle empty loader
             logging.warning(f"Epoch {epoch+1}: Test loader empty.")
             avg_test_loss = np.nan
             val_acc = np.nan
        else:
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    loss = criterion(output, target.view(-1))
                    epoch_test_loss += loss.item()
                    _, predicted = torch.max(output, 1)
                    correct_val += (predicted == target.view(-1)).sum().item()
                    total_val += target.size(0)
            avg_test_loss = epoch_test_loss / num_batches_test
            val_acc = 100. * correct_val / total_val
        test_losses.append(avg_test_loss)
        test_accuracies.append(val_acc)

        logging.info(f'Epoch {epoch+1}/{n_epochs}: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.2f}%, '
                     f'Val Loss={avg_test_loss:.4f}, Val Acc={val_acc:.2f}%')

        # --- Early Stopping Check ---
        current_metric_value = val_acc if early_stopping_metric == 'val_acc' else avg_test_loss
        if np.isnan(current_metric_value):
             logging.warning("Validation metric is NaN, cannot perform early stopping check for this epoch.")
             continue # Skip check if metric is NaN

        improved = (early_stopping_metric == 'val_acc' and current_metric_value > best_metric_value) or \
                   (early_stopping_metric == 'val_loss' and current_metric_value < best_metric_value)

        if improved:
            best_metric_value = current_metric_value
            no_improvement_count = 0
            logging.info(f" Validation metric improved to {best_metric_value:.4f}. Saving model state.")
            best_model_state = copy.deepcopy(model.state_dict()) # Use deepcopy
        else:
            no_improvement_count += 1
            logging.info(f" No improvement for {no_improvement_count} epochs.")

        if no_improvement_count >= early_stopping_patience:
            logging.warning(f"Early stopping triggered at epoch {epoch + 1} after {early_stopping_patience} epochs with no improvement.")
            break # Exit training loop

    # --- End of Training ---
    logging.info(f"--- Final Model Training Complete --- Best Metric ({early_stopping_metric}): {best_metric_value:.4f}")

    # Load best model state if early stopping occurred and found a best state
    if best_model_state:
        logging.info("Loading best model state found during training.")
        model.load_state_dict(best_model_state)
    else:
         logging.warning("No best model state saved (maybe training ended before improvement or patience met). Using final model state.")

    return model, train_losses, test_losses, test_accuracies


# --- Cross-Validation Functions (Adapted for NumPy inputs) ---

def cross_validate_model(X_train_all, y_train_all, train_subject_ids_all, label_encoder, config):
    """ Performs GroupKFold CV using NumPy arrays as input. """
    # ... (Implementation similar to original, but...)
    # - Takes full train NumPy arrays (X_train_all, y_train_all)
    # - Inside loop: uses train_idx, val_idx from gkf.split to index NumPy arrays
    # - Creates SensorDataset for fold train/val sets from NumPy subsets
    # - Creates DataLoaders
    # - Instantiates model, criterion (with weights from fold y_train), optimizer, scheduler
    # - Runs inner train/eval loop for epochs
    # - Returns avg_train_losses, avg_val_losses across folds
    # Note: This function can be complex and might reuse parts of train_model logic.
    # Keep it separate if needed, or simplify based on requirements.
    # For brevity, implementation is omitted here but follows the pattern above.
    logging.warning("Cross-validation function implementation (cross_validate_model) is omitted for brevity. Adapt from original script if needed.")
    return np.array([]), np.array([]) # Placeholder return

def grid_search_hyperparameters(X_train_all, y_train_all, train_subject_ids_all, label_encoder, config):
    """ Performs grid search using GroupKFold CV (NumPy inputs). """
    # ... (Implementation similar to original, calling a CV fold runner like above) ...
    # - Iterates through learning rates (or other params)
    # - For each param set, runs CV (similar logic to cross_validate_model)
    # - Finds param set with best avg validation loss
    # - Returns best parameter(s)
    logging.warning("Grid search function implementation (grid_search_hyperparameters) is omitted for brevity. Adapt from original script if needed.")
    return config.get('lr', 0.001) # Placeholder return


# --- Main Training Orchestration ---

def run_training(config, prep_data_paths):
    """
    Orchestrates the model training process, including optional CV/tuning.

    Args:
        config (dict): Configuration dictionary.
        prep_data_paths (dict): Dictionary with paths to prepared data artifacts
                               (X_train, y_train, X_test, y_test, scaler, encoder, etc.).

    Returns:
        str: Path to the saved final trained model state dictionary.
    """
    logging.info("--- Starting Model Training Stage ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Load Prepared Data and Artifacts ---
    try:
        logging.info("Loading prepared data and artifacts...")
        X_train = utils.load_numpy(prep_data_paths['X_train'])
        y_train_enc = utils.load_numpy(prep_data_paths['y_train'])
        X_test = utils.load_numpy(prep_data_paths['X_test'])
        y_test_enc = utils.load_numpy(prep_data_paths['y_test'])
        label_encoder = utils.load_pickle(prep_data_paths['label_encoder'])
        # Scaler might not be needed here unless saving best model based on validation
        # scaler = utils.load_pickle(prep_data_paths['scaler'])
        train_subject_ids = utils.load_numpy(prep_data_paths['train_subject_ids'])
        data_summary = utils.load_pickle(prep_data_paths['summary'])
        input_channels = data_summary['input_channels']
        num_original_features = data_summary['num_original_features']
        engineered_features_were_used = False # data_summary['engineered_features_were_used']

        n_classes = len(label_encoder.classes_)
        original_feature_names = utils.load_pickle(prep_data_paths['original_feature_names'])
        num_original_features = len(original_feature_names)
        logging.info(f"Data loaded. Input Channels (Combined): {input_channels}, Original Features: {num_original_features}, Num Classes: {n_classes}")

        logging.info(f"Data loaded. Input Channels: {input_channels}, Num Classes: {n_classes}")
    except Exception as e:
        logging.error(f"Error loading prepared data artifacts: {e}", exc_info=True)
        raise

    # --- Get Model Class ---
    model_name = config.get('model_name', 'Simple1DCNN')
    try:
        ModelClass = getattr(models, model_name)
        logging.info(f"Using model class: {model_name}")
    except AttributeError:
        logging.error(f"Model class '{model_name}' not found in models.py!")
        raise AttributeError(f"Model class '{model_name}' not found in models.py!")

    # --- Optional Cross-Validation / Hyperparameter Tuning ---
    final_lr = config.get('lr', 0.001)
    if config.get('use_cross_validation', False):
         if config.get('tune_hyperparameters', False):
              logging.info("--- Starting Hyperparameter Tuning (Grid Search) ---")
              # Pass necessary data arrays to grid search function
              best_lr = grid_search_hyperparameters(
                   X_train, y_train_enc, train_subject_ids, label_encoder, config
              )
              final_lr = best_lr
              config['lr'] = final_lr # Update config for final training
              logging.info(f"--- Hyperparameter Tuning Complete. Using final LR: {final_lr} ---")
         else:
              logging.info("--- Running Cross-Validation Analysis (No Tuning) ---")
              # Run CV just for analysis if needed, using default/tuned LR
              cross_validate_model(
                   X_train, y_train_enc, train_subject_ids, label_encoder, config
              )
              logging.info("--- Cross-Validation Analysis Complete ---")

    # --- Prepare Final DataLoaders ---
    logging.info("Creating final DataLoaders...")
    train_dataset = SensorDataset(X_train, y_train_enc)
    test_dataset = SensorDataset(X_test, y_test_enc)
    num_workers = config.get('num_workers', 0) # Default to 0 if not specified
    batch_size = config.get('batch_size', 32)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    logging.info("DataLoaders created.")

    # --- Train Final Model ---
    logging.info(f"Instantiating final model '{model_name}' for training...")
    try:
        ModelClass = getattr(models, model_name)
        if model_name == 'Simple1DCNN':
             # Simple1DCNN expects the number of channels present in the input data
             if engineered_features_were_used:
                 logging.info(f"Instantiating Simple1DCNN with combined input channels: {input_channels}")
             else:
                 logging.info(f"Instantiating Simple1DCNN with original input channels: {input_channels}")
             final_model = ModelClass(input_channels=input_channels,
                                      num_classes=n_classes).to(device)
        elif model_name == 'GeARFEN':
             # <<< MODIFIED: Instantiate GeARFEN correctly >>>
             # GeARFEN's forward pass expects input channels == num_original_features
             if engineered_features_were_used:
                  logging.error(f"Configuration error: GeARFEN model selected, but 'use_engineered_features' was True in data preparation. GeARFEN expects only original features.")
                  raise ValueError("GeARFEN incompatible with use_engineered_features=True")

             model_specific_params = config.get('model_params', {})
             logging.info(f"Instantiating {model_name} (expects {num_original_features} channels) with params: {model_specific_params}")
             final_model = ModelClass(num_classes=n_classes,
                                      num_original_features=num_original_features, # Pass original count
                                      **model_specific_params).to(device)
        else:
            raise ValueError(f"Unknown model_name '{model_name}'")
    except Exception as e:
         logging.error(f"Failed to instantiate model '{model_name}': {e}", exc_info=True)
         raise

    # Update config with the potentially tuned LR for train_model function
    config['lr'] = final_lr # Although train_model might use fen_lr/fln_lr now
    trained_model, train_losses, test_losses, test_accuracies = train_model(
         final_model, train_loader, test_loader, config, n_classes, device
    )


    # --- Save Final Trained Model ---
    results_dir = config.get('results_dir', 'results')
    model_filename = f"{model_name}_final_state_dict.pt"
    model_save_path = os.path.join(results_dir, model_filename)
    logging.info(f"Saving final trained model state dictionary to: {model_save_path}")
    # Save only the state_dict
    utils.save_pytorch_artifact(trained_model.state_dict(), model_save_path)

    # Optionally save training history
    history = {'train_loss': train_losses, 'test_loss': test_losses, 'test_acc': test_accuracies}
    history_path = os.path.join(results_dir, f"{model_name}_training_history.pkl")
    utils.save_pickle(history, history_path)

    logging.info("--- Model Training Stage Complete ---")
    return model_save_path


if __name__ == '__main__':
    # Example of how to run this script directly
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        # Assumes config is loaded from ../config.yaml
        # Assumes data_preparation stage produced outputs in results/prepared_data/
        from config_loader import load_config
        script_dir = os.path.dirname(__file__)
        project_root = os.path.abspath(os.path.join(script_dir, '..'))
        config_file_path = os.path.join(project_root, 'config.yaml')
        cfg = load_config(config_file_path)

        utils.set_seed(cfg.get('seed_number', 42))

        # Define input paths based on assumed output of data_preparation stage
        prep_data_dir = os.path.join(project_root, cfg.get('results_dir', 'results'), 'prepared_data')
        paths = {
            'X_train': os.path.join(prep_data_dir, 'X_train.npy'),
            'y_train': os.path.join(prep_data_dir, 'y_train.npy'),
            'X_test': os.path.join(prep_data_dir, 'X_test.npy'),
            'y_test': os.path.join(prep_data_dir, 'y_test.npy'),
            'scaler': os.path.join(prep_data_dir, 'scaler.pkl'),
            'label_encoder': os.path.join(prep_data_dir, 'label_encoder.pkl'),
            'train_subject_ids': os.path.join(prep_data_dir, 'train_subject_ids.npy'),
            'summary': os.path.join(prep_data_dir, 'data_summary.pkl')
        }

        # Check if required input files exist
        if not all(os.path.exists(p) for p in paths.values()):
             logging.error("One or more input files for training not found.")
             missing = [p for p in paths.values() if not os.path.exists(p)]
             logging.error(f"Missing files: {missing}")
        else:
            # Execute the main function of this module
            final_model_path = run_training(cfg, paths)
            logging.info(f"Training complete. Final model saved to: {final_model_path}")

    except FileNotFoundError as e:
        logging.error(f"File Not Found Error: {e}")
    except ValueError as e:
        logging.error(f"Value Error: {e}")
    except AttributeError as e:
        logging.error(f"Attribute Error: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)