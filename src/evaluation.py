# src/evaluation.py

import os
import logging
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, precision_score, recall_score, f1_score,
                             roc_curve, auc, roc_auc_score, average_precision_score,
                             precision_recall_curve)
from sklearn.preprocessing import label_binarize

# Assuming utils.py, models.py are in the same src directory or accessible
try:
    from . import utils
    from .utils import SensorDataset # Import Dataset class
    from . import models # Import the models module
except ImportError:
    import utils
    from utils import SensorDataset
    import models # Fallback if run directly

def plot_confusion_matrix(cm, classes, output_path, normalize=False, exclude_class_name=None):
    """Plots and saves the confusion matrix."""
    classes_to_plot = list(classes)
    cm_to_plot = cm.copy()

    if exclude_class_name and exclude_class_name in classes_to_plot:
        try:
            exclude_idx = classes_to_plot.index(exclude_class_name)
            classes_to_plot.pop(exclude_idx)
            cm_to_plot = np.delete(np.delete(cm_to_plot, exclude_idx, axis=0), exclude_idx, axis=1)
            logging.info(f"Excluding class '{exclude_class_name}' from confusion matrix: {output_path}")
        except ValueError:
            logging.warning(f"Class '{exclude_class_name}' not found for exclusion in confusion matrix, plotting all classes.")
        except Exception as e:
            logging.error(f"Error excluding class '{exclude_class_name}' from confusion matrix: {e}")


    plt.figure(figsize=(max(8, len(classes_to_plot)*0.8), max(6, len(classes_to_plot)*0.6)))
    fmt = '.2f' if normalize else 'd'
    title = 'Normalized Confusion Matrix (Recall)' if normalize else 'Confusion Matrix (Counts)'
    if exclude_class_name and exclude_class_name in classes: # Modify title if class was excluded
        title += f" (excluding '{exclude_class_name}')"
    cmap = 'Blues'

    sns.heatmap(cm_to_plot, annot=True, fmt=fmt, cmap=cmap,
                xticklabels=classes_to_plot, yticklabels=classes_to_plot, annot_kws={"size": 8})
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path)
    logging.info(f"Saved confusion matrix plot to '{output_path}'")
    plt.close()

def plot_roc_curve(fpr, tpr, roc_auc, classes, output_path, exclude_class_name=None):
    """Plots and saves the ROC curve."""
    plt.figure(figsize=(10, 8))
    title = 'Receiver Operating Characteristic (One-vs-Rest)'
    if exclude_class_name and exclude_class_name in classes:
        title += f" (excluding '{exclude_class_name}')"

    # Plot each class
    for i in range(len(classes)):
         if classes[i] == exclude_class_name:
             continue
         if i in fpr and i in tpr and i in roc_auc and not np.isnan(roc_auc[i]):
             plt.plot(fpr[i], tpr[i], lw=2, label=f'ROC {classes[i]} (AUC = {roc_auc[i]:.2f})')

    # Plot Micro avg
    if "micro" in fpr and "micro" in tpr and "micro" in roc_auc and not np.isnan(roc_auc['micro']):
        plt.plot(fpr["micro"], tpr["micro"],
                 label=f'Micro-average ROC (AUC = {roc_auc["micro"]:.2f})',
                 color='deeppink', linestyle=':', linewidth=4)
    # Plot Macro avg
    if "macro" in fpr and "macro" in tpr and "macro" in roc_auc and not np.isnan(roc_auc['macro']):
         plt.plot(fpr["macro"], tpr["macro"],
                  label=f'Macro-average ROC (AUC = {roc_auc["macro"]:.2f})',
                  color='navy', linestyle=':', linewidth=4)

    plt.plot([0, 1], [0, 1], 'k--', lw=2) # Random chance line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity/Recall)')
    plt.title(title)
    plt.legend(loc='lower right', fontsize='small')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    logging.info(f"Saved ROC curve plot to '{output_path}'")
    plt.close()

def plot_pr_curve(precision, recall, average_precision, classes, output_path, exclude_class_name=None):
     """Plots and saves the Precision-Recall curve."""
     plt.figure(figsize=(10, 8))
     title = 'Precision-Recall Curve (One-vs-Rest)'
     if exclude_class_name and exclude_class_name in classes:
         title += f" (excluding '{exclude_class_name}')"

     # Plot each class
     for i in range(len(classes)):
          if classes[i] == exclude_class_name:
              continue
          if i in precision and i in recall and i in average_precision and not np.isnan(average_precision[i]):
              plt.plot(recall[i], precision[i], lw=2, label=f'PR {classes[i]} (AP = {average_precision[i]:.2f})')

     # Plot Micro avg
     if "micro" in precision and "micro" in recall and "micro" in average_precision and not np.isnan(average_precision['micro']):
         plt.plot(recall["micro"], precision["micro"],
                  label=f'Micro-average PR (AP = {average_precision["micro"]:.2f})',
                  color='deeppink', linestyle=':', linewidth=4)

     plt.xlabel('Recall (Sensitivity)')
     plt.ylabel('Precision')
     plt.ylim([0.0, 1.05])
     plt.xlim([0.0, 1.0])
     plt.title(title)
     plt.legend(loc='lower left', fontsize='small')
     plt.grid(True)
     plt.tight_layout()
     plt.savefig(output_path)
     logging.info(f"Saved Precision-Recall curve plot to '{output_path}'")
     plt.close()


def evaluate_model(model, test_loader, n_classes, classes, device):
    """
    Performs model inference on the test set and returns predictions, targets, and probabilities.
    """
    model.eval()
    all_predictions = []
    all_targets = []
    all_probs = []

    logging.info("Running inference on test set...")
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            probs = F.softmax(output, dim=1)
            _, predicted = torch.max(output.data, 1)

            all_probs.append(probs.cpu().numpy())
            all_predictions.append(predicted.cpu().numpy())
            all_targets.append(target.cpu().numpy())

    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)
    all_probs = np.concatenate(all_probs)
    logging.info(f"Inference complete. Processed {len(all_targets)} test samples.")

    return all_predictions, all_targets, all_probs


def calculate_metrics(y_true, y_pred, y_prob, n_classes, classes):
    """Calculates various classification metrics."""
    logging.info("Calculating evaluation metrics...")
    metrics = {}

    # Basic Accuracy
    metrics['overall_accuracy'] = (y_pred == y_true).mean()

    # Per-Class Accuracy
    metrics['per_class_accuracy'] = {}
    for i, class_name in enumerate(classes):
        class_mask = (y_true == i)
        class_total = np.sum(class_mask)
        if class_total > 0:
            class_correct = np.sum(y_pred[class_mask] == i)
            metrics['per_class_accuracy'][class_name] = class_correct / class_total
        else:
            metrics['per_class_accuracy'][class_name] = np.nan

    # Weighted & Macro Metrics
    labels_present = np.arange(n_classes) # Assume all classes could be present
    metrics['weighted_precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0, labels=labels_present)
    metrics['weighted_recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0, labels=labels_present)
    metrics['weighted_f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0, labels=labels_present)
    metrics['macro_precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0, labels=labels_present)
    metrics['macro_recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0, labels=labels_present)
    metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0, labels=labels_present)

    # Confusion Matrix
    metrics['confusion_matrix_counts'] = confusion_matrix(y_true, y_pred, labels=labels_present).tolist() # Convert to list for JSON

    # ROC / AUC (One-vs-Rest)
    fpr, tpr, roc_auc = dict(), dict(), dict()
    y_binarized = label_binarize(y_true, classes=labels_present)
    if n_classes == 2 and y_binarized.shape[1] == 1: # Handle binary case as returned by label_binarize
         y_binarized = np.hstack((1 - y_binarized, y_binarized))

    valid_fpr_keys_roc = []
    for i in range(n_classes):
        if i < y_prob.shape[1] and i < y_binarized.shape[1]:
            if len(np.unique(y_binarized[:, i])) > 1: # Check for both classes
                fpr[i], tpr[i], _ = roc_curve(y_binarized[:, i], y_prob[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
                valid_fpr_keys_roc.append(i)
            else: roc_auc[i] = np.nan
        else: roc_auc[i] = np.nan

    # Micro AUC
    try:
        fpr["micro"], tpr["micro"], _ = roc_curve(y_binarized.ravel(), y_prob.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    except Exception as e:
         logging.warning(f"Could not calculate micro-average ROC: {e}")
         roc_auc["micro"] = np.nan

    # Macro AUC
    if valid_fpr_keys_roc:
        all_fpr = np.unique(np.concatenate([fpr[i] for i in valid_fpr_keys_roc]))
        mean_tpr = sum(np.interp(all_fpr, fpr[i], tpr[i]) for i in valid_fpr_keys_roc) / len(valid_fpr_keys_roc)
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    else: roc_auc["macro"] = np.nan

    metrics['roc_auc_per_class'] = {classes[i]: roc_auc.get(i, np.nan) for i in range(n_classes)}
    metrics['roc_auc_micro'] = roc_auc.get("micro", np.nan)
    metrics['roc_auc_macro'] = roc_auc.get("macro", np.nan)
    # Store fpr/tpr for plotting separately if needed
    metrics['_plot_data_roc'] = {'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc}


    # Precision-Recall / Average Precision (One-vs-Rest)
    precision, recall, avg_precision = dict(), dict(), dict()
    valid_ap_keys = []
    for i in range(n_classes):
         if i < y_prob.shape[1] and i < y_binarized.shape[1]:
             if np.sum(y_binarized[:, i]) > 0: # Check for positive samples
                 precision[i], recall[i], _ = precision_recall_curve(y_binarized[:, i], y_prob[:, i])
                 avg_precision[i] = average_precision_score(y_binarized[:, i], y_prob[:, i])
                 valid_ap_keys.append(i)
             else: avg_precision[i] = np.nan
         else: avg_precision[i] = np.nan

    # Micro AP
    try:
        precision["micro"], recall["micro"], _ = precision_recall_curve(y_binarized.ravel(), y_prob.ravel())
        avg_precision["micro"] = average_precision_score(y_binarized, y_prob, average="micro")
    except Exception as e:
        logging.warning(f"Could not calculate micro-average PR/AP: {e}")
        avg_precision["micro"] = np.nan

    # Macro AP
    if valid_ap_keys:
         valid_aps = [avg_precision[i] for i in valid_ap_keys if not np.isnan(avg_precision.get(i))]
         avg_precision["macro"] = np.mean(valid_aps) if valid_aps else np.nan
    else: avg_precision["macro"] = np.nan

    metrics['avg_precision_per_class'] = {classes[i]: avg_precision.get(i, np.nan) for i in range(n_classes)}
    metrics['avg_precision_micro'] = avg_precision.get("micro", np.nan)
    metrics['avg_precision_macro'] = avg_precision.get("macro", np.nan)
    metrics['_plot_data_pr'] = {'precision': precision, 'recall': recall, 'avg_precision': avg_precision}

    logging.info("Metrics calculation complete.")
    return metrics

def run_evaluation(config, model_state_dict_path, prep_data_paths):
    """
    Loads model, data, runs evaluation, calculates metrics, and saves results/plots.
    """
    logging.info("--- Starting Model Evaluation Stage ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = config.get('results_dir', 'results')
    os.makedirs(results_dir, exist_ok=True)

    # --- Load Artifacts ---
    try:
        logging.info("Loading artifacts for evaluation...")
        # Load test data directly
        X_test = utils.load_numpy(prep_data_paths['X_test'])
        y_test_enc = utils.load_numpy(prep_data_paths['y_test'])
        label_encoder = utils.load_pickle(prep_data_paths['label_encoder'])
        data_summary = utils.load_pickle(prep_data_paths['summary'])
        input_channels = data_summary['input_channels'] # Actual channels in X_test.npy
        num_original_features = data_summary['num_original_features']
        engineered_features_were_used = data_summary['engineered_features_used']

        input_channels = data_summary['input_channels']
        n_classes = len(label_encoder.classes_)
        classes = list(label_encoder.classes_)
        original_feature_names = utils.load_pickle(prep_data_paths['original_feature_names'])
        num_original_features = len(original_feature_names)
        # window_size = data_summary['window_size'] # Potentially needed if model depends on it beyond channels

    except Exception as e:
        logging.error(f"Error loading artifacts for evaluation: {e}", exc_info=True)
        raise

    # --- Instantiate Model and Load State ---
    model_name = config.get('model_name', 'Simple1DCNN')
    logging.info(f"Instantiating model '{model_name}' for evaluation...")
    try:
        ModelClass = getattr(models, model_name)
        if model_name == 'Simple1DCNN':
             model = ModelClass(input_channels=input_channels,
                                num_classes=n_classes)
        elif model_name == 'GeARFEN':
            # <<< MODIFIED: Instantiate GeARFEN correctly >>>
            if engineered_features_were_used:
                 # This check might be redundant if training already failed, but good practice
                 logging.error(f"Evaluation config mismatch: GeARFEN model specified, but prepared data used engineered features (input_channels={input_channels}, expected={num_original_features}).")
                 raise ValueError("GeARFEN incompatible with data prepared using engineered features.")
            model_specific_params = config.get('model_params', {})
            logging.info(f"Instantiating {model_name} (expects {num_original_features} channels) with params: {model_specific_params}")
            model = ModelClass(num_classes=n_classes,
                               num_original_features=num_original_features,
                               **model_specific_params)
        else:
             raise ValueError(f"Unknown model_name '{model_name}'")

        logging.info(f"Loading model state dict from: {model_state_dict_path}")
        model.load_state_dict(utils.load_pytorch_artifact(model_state_dict_path, map_location=device))
        model.to(device)
        logging.info(f"Model '{model_name}' loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading model '{model_name}': {e}", exc_info=True)
        raise

    # --- Create Test DataLoader ---
    try:
        test_dataset = SensorDataset(X_test, y_test_enc)
        batch_size = config.get('batch_size', 32)
        num_workers = config.get('num_workers', 0)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        logging.info(f"Test DataLoader created (Batch size: {batch_size}).")
    except Exception as e:
         logging.error(f"Error creating test DataLoader: {e}", exc_info=True)
         raise

    # --- Run Inference ---
    y_pred, y_true, y_prob = evaluate_model(model, test_loader, n_classes, classes, device)

    # --- Calculate Metrics ---
    metrics = calculate_metrics(y_true, y_pred, y_prob, n_classes, classes)

    # --- Log Summary Metrics ---
    logging.info("\n--- Evaluation Metrics Summary ---")
    logging.info(f"Overall Accuracy: {metrics.get('overall_accuracy', np.nan)*100:.2f}%")
    # Log other key metrics like weighted F1, macro F1, AUCs, APs
    logging.info(f"Weighted F1 Score: {metrics.get('weighted_f1', np.nan):.4f}")
    logging.info(f"Macro F1 Score: {metrics.get('macro_f1', np.nan):.4f}")
    logging.info(f"AUC (Micro): {metrics.get('roc_auc_micro', np.nan):.4f}")
    logging.info(f"AUC (Macro): {metrics.get('roc_auc_macro', np.nan):.4f}")
    logging.info(f"Avg Precision (Micro): {metrics.get('avg_precision_micro', np.nan):.4f}")
    logging.info(f"Avg Precision (Macro): {metrics.get('avg_precision_macro', np.nan):.4f}")

    # --- Generate and Save Plots ---
    unknown_class_label = "Unknown" # Assuming this is the label for the unknown class. Adjust if different.
    plot_suffix_no_unknown = f"_no_{unknown_class_label.lower().replace(' ', '_')}"

    try:
        logging.info("Generating and saving plots...")
        # Confusion Matrix (Counts)
        cm_path = os.path.join(results_dir, f"{model_name}_confusion_matrix_counts.png")
        cm_counts = np.array(metrics['confusion_matrix_counts']) # Convert back from list
        plot_confusion_matrix(cm_counts, classes, cm_path, normalize=False)
        # Confusion Matrix (Counts) - No Unknown
        if unknown_class_label in classes:
            cm_path_no_unknown = os.path.join(results_dir, f"{model_name}_confusion_matrix_counts{plot_suffix_no_unknown}.png")
            plot_confusion_matrix(cm_counts, classes, cm_path_no_unknown, normalize=False, exclude_class_name=unknown_class_label)


        # Confusion Matrix (Normalized)
        cm_norm_path = os.path.join(results_dir, f"{model_name}_confusion_matrix_norm.png")
        # Calculate normalized CM for plotting (sum over predicted labels for recall)
        cm_norm = cm_counts.astype('float') / cm_counts.sum(axis=1)[:, np.newaxis]
        np.nan_to_num(cm_norm, copy=False, nan=0.0)
        plot_confusion_matrix(cm_norm, classes, cm_norm_path, normalize=True)
        # Confusion Matrix (Normalized) - No Unknown
        if unknown_class_label in classes:
            cm_norm_path_no_unknown = os.path.join(results_dir, f"{model_name}_confusion_matrix_norm{plot_suffix_no_unknown}.png")
            plot_confusion_matrix(cm_norm, classes, cm_norm_path_no_unknown, normalize=True, exclude_class_name=unknown_class_label)

        # ROC Curve
        roc_plot_data = metrics.get('_plot_data_roc', {})
        if roc_plot_data:
            roc_path = os.path.join(results_dir, f"{model_name}_roc_curve.png")
            plot_roc_curve(roc_plot_data['fpr'], roc_plot_data['tpr'], roc_plot_data['roc_auc'], classes, roc_path)
            # ROC Curve - No Unknown
            if unknown_class_label in classes:
                roc_path_no_unknown = os.path.join(results_dir, f"{model_name}_roc_curve{plot_suffix_no_unknown}.png")
                plot_roc_curve(roc_plot_data['fpr'], roc_plot_data['tpr'], roc_plot_data['roc_auc'], classes, roc_path_no_unknown, exclude_class_name=unknown_class_label)

        # Precision-Recall Curve
        pr_plot_data = metrics.get('_plot_data_pr', {})
        if pr_plot_data:
            pr_path = os.path.join(results_dir, f"{model_name}_pr_curve.png")
            plot_pr_curve(pr_plot_data['precision'], pr_plot_data['recall'], pr_plot_data['avg_precision'], classes, pr_path)
            # PR Curve - No Unknown
            if unknown_class_label in classes:
                pr_path_no_unknown = os.path.join(results_dir, f"{model_name}_pr_curve{plot_suffix_no_unknown}.png")
                plot_pr_curve(pr_plot_data['precision'], pr_plot_data['recall'], pr_plot_data['avg_precision'], classes, pr_path_no_unknown, exclude_class_name=unknown_class_label)

    except Exception as e:
        logging.error(f"Error during plot generation: {e}", exc_info=True)

    # --- Save Metrics ---
    # Remove plot data before saving metrics dictionary
    metrics.pop('_plot_data_roc', None)
    metrics.pop('_plot_data_pr', None)
    metrics_path = os.path.join(results_dir, f"{model_name}_evaluation_metrics.json")
    try:
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer): return int(obj)
            elif isinstance(obj, np.floating): return float(obj)
            elif isinstance(obj, np.ndarray): return obj.tolist()
            elif isinstance(obj, dict): return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list): return [convert_numpy_types(i) for i in obj]
            return obj

        metrics_serializable = convert_numpy_types(metrics)
        with open(metrics_path, 'w') as f:
            json.dump(metrics_serializable, f, indent=4)
        logging.info(f"Saved evaluation metrics to {metrics_path}")
    except Exception as e:
        logging.error(f"Error saving metrics JSON: {e}", exc_info=True)

    logging.info("--- Model Evaluation Stage Complete ---")
    return metrics # Return the calculated metrics


if __name__ == '__main__':
    # Example of how to run this script directly
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        # Assumes config is loaded from ../config.yaml
        # Assumes training stage produced model and data_preparation produced artifacts
        from config_loader import load_config
        script_dir = os.path.dirname(__file__)
        project_root = os.path.abspath(os.path.join(script_dir, '..'))
        config_file_path = os.path.join(project_root, 'config.yaml')
        cfg = load_config(config_file_path)

        utils.set_seed(cfg.get('seed_number', 42)) # Seed for reproducibility if needed

        # Define input paths
        results_dir = os.path.join(project_root, cfg.get('results_dir', 'results'))
        prep_data_dir = os.path.join(results_dir, 'prepared_data')
        model_name_cfg = cfg.get('model_name', 'Simple1DCNN')

        model_path = os.path.join(results_dir, f"{model_name_cfg}_final_state_dict.pt")
        prep_paths = {
            'X_test': os.path.join(prep_data_dir, 'X_test.npy'),
            'y_test': os.path.join(prep_data_dir, 'y_test.npy'),
            'label_encoder': os.path.join(prep_data_dir, 'label_encoder.pkl'),
            'summary': os.path.join(prep_data_dir, 'data_summary.pkl')
        }

        # Check if required input files exist
        required = [model_path] + list(prep_paths.values())
        if not all(os.path.exists(p) for p in required):
             logging.error("One or more input files for evaluation not found.")
             missing = [p for p in required if not os.path.exists(p)]
             logging.error(f"Missing files: {missing}")
        else:
            # Execute the main function of this module
            evaluation_results = run_evaluation(cfg, model_path, prep_paths)
            logging.info("\nEvaluation Results:")
            for key, val in evaluation_results.items():
                if not key.startswith('_plot_data'): # Don't print plot data
                    logging.info(f"  {key}: {val}")

    except FileNotFoundError as e:
        logging.error(f"File Not Found Error: {e}")
    except ValueError as e:
        logging.error(f"Value Error: {e}")
    except AttributeError as e:
        logging.error(f"Attribute Error: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)