# src/data_analyzer.py

import os
import sys
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Add src directory to Python path if needed
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
if project_root not in sys.path:
     sys.path.insert(0, project_root)

try:
    import utils
    import config_loader
except ImportError as e:
     print(f"Error importing utility modules: {e}")
     print("Ensure config_loader.py and utils.py are in the 'src' directory or accessible via PYTHONPATH.")
     sys.exit(1)


def group_features_by_modality(feature_names):
    """Groups feature names based on assumed prefixes."""
    modalities = {}
    # Define known prefixes or patterns - Adjust these based on your actual feature names
    prefixes = [
        'wrist_acc_', 'ear_acc_', 'imu_acc_', 'bioz_acc_', 'gyro_',
        'vivalnk_acc_', 'bottom_value_', 'back_value_'
    ]
    # More generic grouping based on first part before first underscore
    generic_groups = {}
    remaining_features = list(feature_names) # Work on a copy

    # Prioritize specific prefixes
    for prefix in prefixes:
        group = [name for name in remaining_features if name.startswith(prefix)]
        if group:
            modalities[prefix.strip('_')] = group
            remaining_features = [name for name in remaining_features if name not in group] # Remove assigned features

    # Group remaining features by first part of name
    for name in remaining_features:
        base = name.split('_')[0]
        if base not in generic_groups:
            generic_groups[base] = []
        generic_groups[base].append(name)

    modalities.update(generic_groups) # Add remaining groups
    logging.info(f"Grouped features into modalities: {list(modalities.keys())}")
    return modalities


def display_window_analysis(config,
                              x_windows_raw_path,
                              engineered_features_path,
                              y_windows_path,
                              subject_ids_path,
                              window_start_times_path,
                              window_end_times_path,
                              original_feature_names_path,
                              engineered_names_path):
    """
    Interactively loads and displays data for a selected window.

    Args:
        config (dict): Configuration dictionary (for parameters like sampling rate).
        x_windows_raw_path (str): Path to raw feature windows (.npy).
        engineered_features_path (str): Path to engineered features (.npy).
        y_windows_path (str): Path to window labels (.npy).
        original_feature_names_path (str): Path to list of original feature names (.pkl).
        engineered_names_path (str): Path to list of engineered feature names (.pkl).
    """
    logging.info("--- Starting Data Analyzer ---")

    # --- Load Data ---
    try:
        logging.info("Loading data for analysis...")
        X_windows_raw = utils.load_numpy(x_windows_raw_path)
        engineered_features = utils.load_numpy(engineered_features_path)
        y_windows = utils.load_numpy(y_windows_path, allow_pickle=True)
        original_feature_names = utils.load_pickle(original_feature_names_path)
        engineered_feature_names = utils.load_pickle(engineered_names_path)
        subject_ids_windows = utils.load_numpy(subject_ids_path, allow_pickle=True)
        window_start_times = utils.load_numpy(window_start_times_path)
        window_end_times = utils.load_numpy(window_end_times_path)
        logging.info(f"Data loaded. Number of windows: {X_windows_raw.shape[0]}")

        feature_groups = group_features_by_modality(original_feature_names)
        # Create mapping from full name to index for easier slicing
        original_name_to_index = {name: idx for idx, name in enumerate(original_feature_names)}

        # Verify dimensions
        if not (X_windows_raw.shape[0] == engineered_features.shape[0] == y_windows.shape[0] == subject_ids_windows.shape[0]):
            raise ValueError("Mismatch in number of windows between loaded arrays (raw, eng, labels, subjects).")
        if X_windows_raw.shape[2] != len(original_feature_names):
            logging.warning("Mismatch between raw features dim and original names list.")
        if engineered_features.shape[1] != len(engineered_feature_names):
            logging.warning("Mismatch between engineered features dim and engineered names list.")
        if not (X_windows_raw.shape[0] == engineered_features.shape[0] == y_windows.shape[0] ==
                subject_ids_windows.shape[0] == window_start_times.shape[0] == window_end_times.shape[0]):
            raise ValueError("Mismatch in number of windows between loaded arrays.")


    except FileNotFoundError as e:
        logging.error(f"Error loading data: {e}. Please ensure feature_engineering stage ran successfully.")
        return
    except Exception as e:
        logging.error(f"An unexpected error occurred during data loading: {e}", exc_info=True)
        return

    n_windows, window_size, n_original_features = X_windows_raw.shape
    n_engineered_features = engineered_features.shape[1]
    sampling_rate = config.get('downsample_freq', 20) # Get sampling rate for time axis
    time_axis = np.arange(window_size) / sampling_rate # Time axis in seconds

    # --- Interactive Loop ---
    while True:
        try:
            user_input = input(f"\nEnter window index (0 to {n_windows-1}) or 'q' to quit: ").strip().lower()
            if user_input in ['q', 'quit', 'exit']:
                break

            window_index = int(user_input)
            if not (0 <= window_index < n_windows):
                print(f"Error: Index must be between 0 and {n_windows-1}.")
                continue

            # --- Select Data for the Window ---
            raw_window = X_windows_raw[window_index] # Shape (window_size, n_original_features)
            eng_features_vec = engineered_features[window_index] # Shape (n_engineered_features,)
            label = y_windows[window_index]
            subject_id = subject_ids_windows[window_index]
            start_time = pd.to_datetime(window_start_times[window_index]) # Convert from numpy datetime64
            end_time = pd.to_datetime(window_end_times[window_index])


            print("-" * 50)
            print(f"--- Analyzing Window Index: {window_index} ---")
            print(f"  Subject ID: {subject_id}")
            print(f"  Label:      {label}")
            print(f"  Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}") # Format ms
            print(f"  End Time:   {end_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")   # Format ms
            print("-" * 50)

            # --- Calculate and Display Statistical Analysis per Modality ---
            print("\nStatistical Analysis (Raw Window Data by Modality):")
            stats_results = {}
            for modality_name, feature_list in feature_groups.items():
                modality_indices = [original_name_to_index.get(name) for name in feature_list if name in original_name_to_index]
                if not modality_indices:
                     logging.debug(f"Skipping modality '{modality_name}' - no features found in data.")
                     continue

                modality_data = raw_window[:, modality_indices] # Shape (window_size, n_modality_features)

                # Calculate stats across time axis (axis=0)
                means = np.mean(modality_data, axis=0)
                stds = np.std(modality_data, axis=0)
                mins = np.min(modality_data, axis=0)
                maxs = np.max(modality_data, axis=0)
                medians = np.median(modality_data, axis=0)

                # Create a small DataFrame for nice printing
                stats_df = pd.DataFrame({
                    'Mean': means,
                    'StdDev': stds,
                    'Min': mins,
                    'Max': maxs,
                    'Median': medians,
                    'Range': maxs - mins
                }, index=[original_feature_names[i] for i in modality_indices]) # Use names as index

                stats_results[modality_name] = stats_df
                print(f"\n  --- Modality: {modality_name} ---")
                print(stats_df.to_string(float_format="%.4f")) # Print full DataFrame

            print("-" * 70)


            # --- Display Engineered Features ---
            print("\nEngineered Features:")
            if n_engineered_features == len(engineered_feature_names):
                max_name_len = max(len(name) for name in engineered_feature_names) if engineered_feature_names else 0
                for i in range(n_engineered_features):
                    print(f"  {engineered_feature_names[i]:<{max_name_len}} : {eng_features_vec[i]:.4f}")
            else: # Fallback if names list doesn't match
                 print("Warning: Mismatch between number of engineered features and names.")
                 print(eng_features_vec)
            print("-" * 50)

            # --- Plot Raw Window Data ---
            print("\nGenerating plot for raw sensor data in the window...")
            fig, ax = plt.subplots(figsize=(15, 7))

            if n_original_features == len(original_feature_names):
                for i in range(n_original_features):
                    ax.plot(time_axis, raw_window[:, i], label=original_feature_names[i], alpha=0.8)
            else: # Fallback if names list doesn't match
                print("Warning: Mismatch between number of raw features and names. Plotting without labels.")
                for i in range(n_original_features):
                     ax.plot(time_axis, raw_window[:, i], alpha=0.8)


            title = (f"Raw Sensor Data - Window {window_index} (Subject: {subject_id}, Label: {label})\n"
                     f"Time: {start_time.strftime('%H:%M:%S.%f')[:-3]} to {end_time.strftime('%H:%M:%S.%f')[:-3]}")
            ax.set_title(title)
            ax.set_xlabel(f"Time within window (seconds, @{sampling_rate}Hz)")
            ax.set_ylabel("Sensor Value")
            ax.grid(True)
            # Shrink plot slightly to fit legend below/beside if too many features
            if n_original_features > 10:
                 box = ax.get_position()
                 ax.set_position([box.x0, box.y0 + box.height * 0.15,
                                  box.width, box.height * 0.85])
                 ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
                           fancybox=True, shadow=True, ncol=min(n_original_features // 2, 5)) # Adjust ncol
            else:
                ax.legend()

            plt.tight_layout(rect=[0, 0.05, 1, 1]) # Adjust layout slightly if legend is outside
            # Save the plot to a file
            plot_filename = os.path.join(feature_dir, f"window_{window_index}_raw_plot.png")
            plt.savefig(plot_filename)
            print(f"Plot saved to {plot_filename}")
            # Show the plot
            plt.show() # Display the plot (blocks until closed)

            print("Plot window closed.")

        except ValueError:
            print("Invalid input. Please enter a number or 'q'.")
        except KeyboardInterrupt:
             print("\nExiting analyzer.")
             break
        except Exception as e:
            logging.error(f"An error occurred during analysis: {e}", exc_info=True)
            # Optionally break or continue
            # break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze specific windows of processed sensor data.")
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to the YAML configuration file.')
    parser.add_argument('--feature-dir', type=str, default=None,
                        help='Path to the directory containing feature engineering outputs (overrides config).')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set the logging level.')

    args = parser.parse_args()

    # Setup Logging
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler()]) # Log only to console for analyzer

    # Load Config
    try:
        config_path = args.config
        if not os.path.isabs(config_path):
            config_path = os.path.join(project_root, config_path)
        cfg = config_loader.load_config(config_path)
    except Exception as e:
        logging.error(f"Failed to load config file '{args.config}': {e}")
        sys.exit(1)

    # Determine feature directory path
    feature_dir_arg = args.feature_dir
    if feature_dir_arg and not os.path.isabs(feature_dir_arg):
        feature_dir_arg = os.path.join(project_root, feature_dir_arg)

    feature_dir = feature_dir_arg or os.path.join(project_root, cfg.get('intermediate_feature_dir', 'features'))
    logging.info(f"Using feature directory: {feature_dir}")

    # Define input file paths
    x_raw_path = os.path.join(feature_dir, 'X_windows_raw.npy')
    eng_feat_path = os.path.join(feature_dir, 'engineered_features.npy')
    y_win_path = os.path.join(feature_dir, 'y_windows.npy')
    orig_names_path = os.path.join(feature_dir, 'original_feature_names.pkl')
    eng_names_path = os.path.join(feature_dir, 'engineered_feature_names.pkl')
    subj_ids_path = os.path.join(feature_dir, 'subject_ids_windows.npy')
    start_times_path = os.path.join(feature_dir, 'window_start_times.npy')
    end_times_path = os.path.join(feature_dir, 'window_end_times.npy')


    # Check required files exist
    required = [x_raw_path, eng_feat_path, y_win_path, orig_names_path,
                eng_names_path, subj_ids_path, start_times_path, end_times_path]
    if not all(os.path.exists(p) for p in required):
        logging.error("One or more required input files (.npy, .pkl) not found in the feature directory.")
        missing = [p for p in required if not os.path.exists(p)]
        logging.error(f"Missing files: {missing}")
        logging.error("Please ensure the 'feature_engineering' stage has been run successfully.")
        sys.exit(1)

    # Run the analysis tool
    display_window_analysis(
        config=cfg,
        x_windows_raw_path=x_raw_path,
        engineered_features_path=eng_feat_path,
        y_windows_path=y_win_path,
        subject_ids_path=subj_ids_path,
        window_start_times_path=start_times_path, # <<< ADDED
        window_end_times_path=end_times_path,     # <<< ADDED
        original_feature_names_path=orig_names_path,
        engineered_names_path=eng_names_path
    )

    logging.info("--- Data Analyzer Finished ---")