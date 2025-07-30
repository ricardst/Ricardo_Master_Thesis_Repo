"""
Data Exploration Script for Activity Classification
Analyzes window-level and subject-level statistics with comprehensive visualizations.
Optimized for parallel processing using multiprocessing.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import yaml
from pathlib import Path
import warnings
from multiprocessing import Pool, cpu_count
from functools import partial
import pickle
from tqdm import tqdm
import json

warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_config(config_path="config.yaml"):
    """Load configuration file"""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Config file {config_path} not found. Using default values.")
        return {
            'intermediate_feature_dir': 'results/pipeline',
            'project_root': os.getcwd(),
            'excluded_subjects_manual': ['OutSense-036', 'OutSense-425', 'OutSense-515'],
            'selected_classes': ['Propulsion', 'Resting', 'Transfer', 'Exercising', 'Conversation'],
            'min_class_instances': 10
        }

def get_feature_names_mapping(config):
    """Create mapping from feature indices to feature names"""
    # Try to get sensor columns from pipeline metadata first
    pipeline_results_dir = os.path.join(config.get('project_root', os.getcwd()), 'results/pipeline')
    sensor_config_path = os.path.join(pipeline_results_dir, 'sensor_configuration.json')
    
    if os.path.exists(sensor_config_path):
        try:
            with open(sensor_config_path, 'r') as f:
                sensor_config = json.load(f)
            sensor_columns = sensor_config.get('final_sensor_columns', [])
            print(f"Loaded sensor columns from pipeline configuration: {len(sensor_columns)} sensors")
        except Exception as e:
            print(f"Error loading sensor configuration: {e}")
            sensor_columns = []
    else:
        # Fallback to config file
        sensor_columns = config.get('sensor_columns_original', [])
    
    if not sensor_columns:
        print("Warning: No sensor columns found. Using generic feature names.")
        return {}
    
    # Create mapping: feature_idx -> feature_name
    feature_mapping = {i: name for i, name in enumerate(sensor_columns)}
    
    print(f"Created feature mapping for {len(feature_mapping)} features")
    return feature_mapping

def format_feature_name(feature_idx, feature_mapping, max_length=20):
    """Format feature name for display, with optional truncation"""
    if feature_idx in feature_mapping:
        name = feature_mapping[feature_idx]
        if len(name) > max_length:
            return name[:max_length-3] + "..."
        return name
    else:
        return f"Feature_{feature_idx}"

def create_feature_summary_with_names(window_df, feature_mapping):
    """Create a comprehensive feature summary with names"""
    feature_stats = window_df.groupby('feature_idx').agg({
        'mean': ['mean', 'std', 'min', 'max'],
        'std': ['mean', 'std'],
        'skewness': ['mean', 'std'],
        'kurtosis': ['mean', 'std'],
        'range': ['mean', 'std'],
        'iqr': ['mean', 'std']
    }).round(4)
    
    # Flatten column names
    feature_stats.columns = ['_'.join(col).strip() for col in feature_stats.columns]
    feature_stats = feature_stats.reset_index()
    
    # Add feature names
    feature_stats['feature_name'] = feature_stats['feature_idx'].map(
        lambda x: feature_mapping.get(x, f"Feature_{x}")
    )
    
    # Reorder columns to put name first
    cols = ['feature_idx', 'feature_name'] + [col for col in feature_stats.columns if col not in ['feature_idx', 'feature_name']]
    feature_stats = feature_stats[cols]
    
    return feature_stats

def prepare_all_data_for_subject_optimization(config, project_root, fixed_test_subjects=None):
    """Load all available data from pipeline output files with filtering"""
    print("--- Starting Data Preparation from Pipeline Output ---")
    
    # Update paths to use pipeline output files
    pipeline_results_dir = os.path.join(project_root, 'results/pipeline')
    combined_dataset_path = os.path.join(pipeline_results_dir, "combined_windowed_dataset_mapped_filtered.pkl")
    metadata_path = os.path.join(pipeline_results_dir, "combined_dataset_metadata_mapped_filtered.json")

    # Check if pipeline files exist
    if not os.path.exists(combined_dataset_path):
        print(f"ERROR: Pipeline dataset file not found: {combined_dataset_path}")
        print("Please run pipeline.ipynb first to create the windowed dataset.")
        return None

    try:
        # Load the combined dataset from pipeline
        print(f"Loading combined dataset from: {combined_dataset_path}")
        with open(combined_dataset_path, 'rb') as f:
            combined_dataset = pickle.load(f)
        
        # Extract data components from pipeline structure
        X_windows = combined_dataset['windows']
        y_windows = combined_dataset['labels']
        window_info = combined_dataset['window_info']
        
        # Extract subject ID from window_info
        subject_column = 'SubjectID'  # Default from pipeline
        if subject_column not in window_info.columns:
            # Try alternative column names
            possible_columns = ['subject_id', 'SubjectID', 'Subject', 'subject']
            for col in possible_columns:
                if col in window_info.columns:
                    subject_column = col
                    break
        
        if subject_column not in window_info.columns:
            print(f"ERROR: No subject ID column found in window_info")
            print(f"Available columns: {list(window_info.columns)}")
            return None
        
        subject_ids_windows = window_info[subject_column].values
        
        print(f"Data loaded successfully from pipeline:")
        print(f"  X_windows shape: {X_windows.shape}")
        print(f"  y_windows shape: {y_windows.shape}")
        print(f"  subject_ids_windows shape: {subject_ids_windows.shape}")
        
        # Load metadata if available
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            print(f"  Loaded metadata: {metadata.get('n_subjects', 'N/A')} subjects, {metadata.get('n_unique_labels', 'N/A')} labels")
        
    except Exception as e:
        print(f"Error loading data from pipeline files: {e}")
        return None

    # The pipeline data should already be mapped and filtered, so we can skip label remapping
    print("INFO: Using pre-processed data from pipeline (already mapped and filtered)")
    
    # Manual subject exclusion (if any subjects not already filtered)
    excluded_subjects_manual = config.get('excluded_subjects_manual', [])
    if excluded_subjects_manual:
        print(f"Checking for manual subject exclusions: {excluded_subjects_manual}")
        initial_count = len(subject_ids_windows)
        exclusion_mask = ~np.isin(subject_ids_windows, excluded_subjects_manual)
        if not exclusion_mask.all():
            X_windows = X_windows[exclusion_mask]
            y_windows = y_windows[exclusion_mask]
            subject_ids_windows = subject_ids_windows[exclusion_mask]
            excluded_count = initial_count - len(subject_ids_windows)
            print(f"Manually excluded {excluded_count} additional data points.")
        else:
            print(f"No additional subjects to exclude (may have been filtered in pipeline).")

    # Additional filtering for minimum class instances (if needed)
    min_instances_threshold = config.get('min_class_instances', 10)
    if min_instances_threshold > 0 and len(y_windows) > 0:
        print(f"INFO: Checking minimum class instances threshold: {min_instances_threshold}")
        unique_labels, counts = np.unique(y_windows, return_counts=True)
        labels_to_keep = unique_labels[counts >= min_instances_threshold]
        if len(labels_to_keep) < len(unique_labels):
            instance_mask = np.isin(y_windows, labels_to_keep)
            X_windows = X_windows[instance_mask]
            y_windows = y_windows[instance_mask]
            subject_ids_windows = subject_ids_windows[instance_mask]
            print(f"  Removed {len(unique_labels) - len(labels_to_keep)} classes with insufficient instances.")
        else:
            print(f"  All classes meet the minimum instance threshold.")

    # Convert to float32 to reduce memory usage
    if X_windows.dtype != np.float32:
        print(f"INFO: Converting X_windows from {X_windows.dtype} to np.float32.")
        X_windows = X_windows.astype(np.float32)

    print(f"Final data: {X_windows.shape[0]} windows, {X_windows.shape[1]} timesteps, {X_windows.shape[2]} features")
    print(f"Unique subjects: {np.unique(subject_ids_windows)}")
    print(f"Unique labels: {np.unique(y_windows)}")
    
    return X_windows, y_windows, subject_ids_windows

def process_subject_windows_parallel(args):
    """Process window statistics for a single subject - designed for multiprocessing"""
    subject_id, subject_mask, X_windows, y_windows, subject_ids_windows = args
    
    try:
        # Get data for this subject
        subject_X = X_windows[subject_mask]
        subject_y = y_windows[subject_mask]
        subject_window_indices = np.where(subject_mask)[0]
        
        n_subject_windows, n_timesteps, n_features = subject_X.shape
        
        # Calculate statistics for each window and feature
        window_stats = []
        
        for local_window_idx in range(n_subject_windows):
            global_window_idx = subject_window_indices[local_window_idx]
            
            for feature_idx in range(n_features):
                window_data = subject_X[local_window_idx, :, feature_idx]
                
                # Basic statistics
                mean_val = np.mean(window_data)
                std_val = np.std(window_data)
                min_val = np.min(window_data)
                max_val = np.max(window_data)
                median_val = np.median(window_data)
                q25 = np.percentile(window_data, 25)
                q75 = np.percentile(window_data, 75)
                
                # Advanced statistics
                skew_val = stats.skew(window_data)
                kurt_val = stats.kurtosis(window_data)
                range_val = max_val - min_val
                iqr_val = q75 - q25
                
                # Store statistics
                stats_dict = {
                    'window_id': global_window_idx,
                    'subject_id': subject_id,
                    'label': subject_y[local_window_idx],
                    'feature_idx': feature_idx,
                    'mean': mean_val,
                    'std': std_val,
                    'min': min_val,
                    'max': max_val,
                    'median': median_val,
                    'q25': q25,
                    'q75': q75,
                    'skewness': skew_val,
                    'kurtosis': kurt_val,
                    'range': range_val,
                    'iqr': iqr_val
                }
                window_stats.append(stats_dict)
        
        return subject_id, window_stats
        
    except Exception as e:
        print(f"Error processing subject {subject_id}: {e}")
        return subject_id, []

def process_subject_aggregates_parallel(args):
    """Process subject-level aggregated statistics - designed for multiprocessing"""
    subject_id, subject_mask, X_windows, y_windows, subject_ids_windows = args
    
    try:
        # Get data for this subject
        subject_X = X_windows[subject_mask]
        subject_y = y_windows[subject_mask]
        
        n_subject_windows, n_timesteps, n_features = subject_X.shape
        
        # Calculate label distribution for this subject
        unique_labels, label_counts = np.unique(subject_y, return_counts=True)
        label_dist = dict(zip(unique_labels, label_counts))
        
        # Calculate aggregated statistics for each feature
        subject_stats = []
        
        for feature_idx in range(n_features):
            # Extract all data for this feature across all windows for this subject
            feature_data = subject_X[:, :, feature_idx].flatten()
            
            # Calculate window-level means and stds for this subject and feature
            window_means = np.mean(subject_X[:, :, feature_idx], axis=1)
            window_stds = np.std(subject_X[:, :, feature_idx], axis=1)
            
            # Store statistics
            stats_dict = {
                'subject_id': subject_id,
                'feature_idx': feature_idx,
                'n_windows': n_subject_windows,
                'mean_of_means': np.mean(window_means),
                'std_of_means': np.std(window_means),
                'mean_of_stds': np.mean(window_stds),
                'overall_mean': np.mean(feature_data),
                'overall_std': np.std(feature_data),
                'overall_min': np.min(feature_data),
                'overall_max': np.max(feature_data),
                'overall_median': np.median(feature_data),
                'label_distribution': str(label_dist)
            }
            subject_stats.append(stats_dict)
        
        return subject_id, subject_stats
        
    except Exception as e:
        print(f"Error processing subject aggregates {subject_id}: {e}")
        return subject_id, []

def analyze_window_statistics_parallel(X_windows, y_windows, subject_ids_windows, n_jobs=4):
    """Analyze window-level statistics using parallel processing"""
    print(f"\n--- Window-Level Analysis (Parallel with {n_jobs} processes) ---")
    
    # Get unique subjects
    unique_subjects = np.unique(subject_ids_windows)
    n_windows, n_timesteps, n_features = X_windows.shape
    
    print(f"Analyzing {n_windows} windows with {n_timesteps} timesteps and {n_features} features")
    print(f"Processing {len(unique_subjects)} subjects: {unique_subjects}")
    
    # Prepare arguments for parallel processing
    args_list = []
    for subject_id in unique_subjects:
        subject_mask = subject_ids_windows == subject_id
        args_list.append((subject_id, subject_mask, X_windows, y_windows, subject_ids_windows))
    
    # Process in parallel
    all_window_stats = []
    
    with Pool(processes=n_jobs) as pool:
        # Use tqdm for progress bar
        results = list(tqdm(
            pool.imap(process_subject_windows_parallel, args_list),
            total=len(args_list),
            desc="Processing subjects (windows)"
        ))
    
    # Combine results
    for subject_id, subject_window_stats in results:
        all_window_stats.extend(subject_window_stats)
    
    print(f"Window statistics calculated for {len(all_window_stats)} window-feature combinations")
    return pd.DataFrame(all_window_stats)

def analyze_subject_statistics_parallel(X_windows, y_windows, subject_ids_windows, n_jobs=4):
    """Analyze subject-level statistics using parallel processing"""
    print(f"\n--- Subject-Level Analysis (Parallel with {n_jobs} processes) ---")
    
    unique_subjects = np.unique(subject_ids_windows)
    n_features = X_windows.shape[2]
    
    print(f"Analyzing {len(unique_subjects)} subjects: {unique_subjects}")
    
    # Prepare arguments for parallel processing
    args_list = []
    for subject_id in unique_subjects:
        subject_mask = subject_ids_windows == subject_id
        args_list.append((subject_id, subject_mask, X_windows, y_windows, subject_ids_windows))
    
    # Process in parallel
    all_subject_stats = []
    
    with Pool(processes=n_jobs) as pool:
        # Use tqdm for progress bar
        results = list(tqdm(
            pool.imap(process_subject_aggregates_parallel, args_list),
            total=len(args_list),
            desc="Processing subjects (aggregates)"
        ))
    
    # Combine results
    for subject_id, subject_stats in results:
        all_subject_stats.extend(subject_stats)
    
    print(f"Subject statistics calculated for {len(all_subject_stats)} subject-feature combinations")
    return pd.DataFrame(all_subject_stats)

def visualize_window_statistics(window_stats_df, output_dir="data_exploration", config=None):
    """Create comprehensive visualizations for window-level statistics with feature names"""
    print("\n--- Creating Window-Level Visualizations ---")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get feature mapping
    feature_mapping = get_feature_names_mapping(config) if config else {}
    
    # Sample data if too large for visualization
    if len(window_stats_df) > 500000000:
        print(f"Sampling 50000 windows from {len(window_stats_df)} for visualization...")
        window_sample = window_stats_df.sample(n=50000, random_state=42)
    else:
        window_sample = window_stats_df
    
    # Get all unique features
    unique_features = sorted(window_sample['feature_idx'].unique())
    n_features = len(unique_features)
    print(f"Creating visualizations for all {n_features} features...")
    
    # 1. COMPREHENSIVE OVERVIEW PLOTS WITH ALL FEATURES
    plt.figure(figsize=(20, 15))
    
    # Distribution of means across features
    plt.subplot(2, 3, 1)
    feature_means = window_sample.groupby('feature_idx')['mean'].mean()
    feature_names = [format_feature_name(idx, feature_mapping, 15) for idx in feature_means.index]
    plt.bar(range(len(feature_means)), feature_means.values)
    plt.title(f'Average Window Means by Feature (All {n_features} Features)')
    plt.xlabel('Feature')
    plt.ylabel('Average Mean')
    plt.xticks(range(0, len(feature_names), max(1, len(feature_names)//10)), 
               [feature_names[i] for i in range(0, len(feature_names), max(1, len(feature_names)//10))], 
               rotation=45, ha='right')
    
    # Feature variability 
    plt.subplot(2, 3, 2)
    feature_stds = window_sample.groupby('feature_idx')['std'].mean()
    feature_names_std = [format_feature_name(idx, feature_mapping, 15) for idx in feature_stds.index]
    plt.bar(range(len(feature_stds)), feature_stds.values)
    plt.title(f'Average Standard Deviation by Feature (All {n_features} Features)')
    plt.xlabel('Feature')
    plt.ylabel('Average Std')
    plt.xticks(range(0, len(feature_names_std), max(1, len(feature_names_std)//10)), 
               [feature_names_std[i] for i in range(0, len(feature_names_std), max(1, len(feature_names_std)//10))], 
               rotation=45, ha='right')
    
    # Mean vs Standard Deviation scatter (colored by feature)
    plt.subplot(2, 3, 3)
    scatter = plt.scatter(window_sample['mean'], window_sample['std'], 
                         c=window_sample['feature_idx'], alpha=0.5, s=1, cmap='tab20')
    plt.xlabel('Mean')
    plt.ylabel('Standard Deviation')
    plt.title('Mean vs Std Deviation (colored by feature)')
    cbar = plt.colorbar(scatter, label='Feature Index')
    # Add some feature name examples to the colorbar if there aren't too many features
    if n_features <= 20:
        tick_positions = np.linspace(0, n_features-1, min(5, n_features))
        tick_labels = [format_feature_name(int(pos), feature_mapping, 8) for pos in tick_positions]
        cbar.set_ticks(tick_positions)
        cbar.set_ticklabels(tick_labels)
    
    # Skewness by feature (ALL features)
    plt.subplot(2, 3, 4)
    feature_skew = window_sample.groupby('feature_idx')['skewness'].mean()
    feature_names_skew = [format_feature_name(idx, feature_mapping, 15) for idx in feature_skew.index]
    plt.bar(range(len(feature_skew)), feature_skew.values)
    plt.title(f'Average Skewness by Feature (All {n_features} Features)')
    plt.xlabel('Feature')
    plt.ylabel('Average Skewness')
    plt.xticks(range(0, len(feature_names_skew), max(1, len(feature_names_skew)//10)), 
               [feature_names_skew[i] for i in range(0, len(feature_names_skew), max(1, len(feature_names_skew)//10))], 
               rotation=45, ha='right')
    
    # Range by feature (ALL features)
    plt.subplot(2, 3, 5)
    feature_range = window_sample.groupby('feature_idx')['range'].mean()
    feature_names_range = [format_feature_name(idx, feature_mapping, 15) for idx in feature_range.index]
    plt.bar(range(len(feature_range)), feature_range.values)
    plt.title(f'Average Range by Feature (All {n_features} Features)')
    plt.xlabel('Feature')
    plt.ylabel('Average Range')
    plt.xticks(range(0, len(feature_names_range), max(1, len(feature_names_range)//10)), 
               [feature_names_range[i] for i in range(0, len(feature_names_range), max(1, len(feature_names_range)//10))], 
               rotation=45, ha='right')
    
    # Distribution by label
    plt.subplot(2, 3, 6)
    sns.boxplot(data=window_sample, x='label', y='mean')
    plt.title('Mean Distribution by Label')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'window_statistics_overview_all_features.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. COMPREHENSIVE FEATURE ANALYSIS (ALL FEATURES IN ONE PLOT)
    feature_stats = window_sample.groupby('feature_idx').agg({
        'mean': ['mean', 'std', 'min', 'max'],
        'std': ['mean', 'std'],
        'skewness': 'mean',
        'kurtosis': 'mean',
        'range': 'mean',
        'iqr': 'mean'
    }).round(4)
    
    # Flatten column names
    feature_stats.columns = ['_'.join(col).strip() for col in feature_stats.columns]
    feature_stats = feature_stats.reset_index()
    
    # Create comprehensive feature analysis plot
    plt.figure(figsize=(20, 16))
    
    # Plot 1: Average means for ALL features
    plt.subplot(3, 2, 1)
    feature_names_short = [format_feature_name(idx, feature_mapping, 8) for idx in feature_stats['feature_idx']]
    plt.bar(range(len(feature_stats)), feature_stats['mean_mean'])
    plt.title(f'Average Mean by Feature (All {n_features} Features)')
    plt.xlabel('Feature')
    plt.ylabel('Average Mean')
    plt.xticks(range(0, len(feature_names_short), max(1, len(feature_names_short)//15)), 
               [feature_names_short[i] for i in range(0, len(feature_names_short), max(1, len(feature_names_short)//15))], 
               rotation=45, ha='right')
    
    # Plot 2: Variability of means for ALL features
    plt.subplot(3, 2, 2)
    feature_names_var = [format_feature_name(idx, feature_mapping, 8) for idx in feature_stats['feature_idx']]
    plt.bar(range(len(feature_stats)), feature_stats['mean_std'])
    plt.title(f'Variability of Means by Feature (All {n_features} Features)')
    plt.xlabel('Feature')
    plt.ylabel('Std of Means')
    plt.xticks(range(0, len(feature_names_var), max(1, len(feature_names_var)//15)), 
               [feature_names_var[i] for i in range(0, len(feature_names_var), max(1, len(feature_names_var)//15))], 
               rotation=45, ha='right')
    
    # Plot 3: Average standard deviation for ALL features
    plt.subplot(3, 2, 3)
    feature_names_std_avg = [format_feature_name(idx, feature_mapping, 8) for idx in feature_stats['feature_idx']]
    plt.bar(range(len(feature_stats)), feature_stats['std_mean'])
    plt.title(f'Average Standard Deviation by Feature (All {n_features} Features)')
    plt.xlabel('Feature')
    plt.ylabel('Average Std')
    step_size = max(1, len(feature_names_std_avg)//15)
    plt.xticks(range(0, len(feature_names_std_avg), step_size), 
               [feature_names_std_avg[i] for i in range(0, len(feature_names_std_avg), step_size)], 
               rotation=45, ha='right')
    
    # Plot 4: Average skewness for ALL features
    plt.subplot(3, 2, 4)
    feature_names_skew_avg = [format_feature_name(idx, feature_mapping, 8) for idx in feature_stats['feature_idx']]
    plt.bar(range(len(feature_stats)), feature_stats['skewness_mean'])
    plt.title(f'Average Skewness by Feature (All {n_features} Features)')
    plt.xlabel('Feature')
    plt.ylabel('Average Skewness')
    step_size = max(1, len(feature_names_skew_avg)//15)
    plt.xticks(range(0, len(feature_names_skew_avg), step_size), 
               [feature_names_skew_avg[i] for i in range(0, len(feature_names_skew_avg), step_size)], 
               rotation=45, ha='right')
    
    # Plot 5: Average kurtosis for ALL features
    plt.subplot(3, 2, 5)
    feature_names_kurt = [format_feature_name(idx, feature_mapping, 8) for idx in feature_stats['feature_idx']]
    plt.bar(range(len(feature_stats)), feature_stats['kurtosis_mean'])
    plt.title(f'Average Kurtosis by Feature (All {n_features} Features)')
    plt.xlabel('Feature')
    plt.ylabel('Average Kurtosis')
    step_size = max(1, len(feature_names_kurt)//15)
    plt.xticks(range(0, len(feature_names_kurt), step_size), 
               [feature_names_kurt[i] for i in range(0, len(feature_names_kurt), step_size)], 
               rotation=45, ha='right')
    
    # Plot 6: Range and IQR comparison
    plt.subplot(3, 2, 6)
    x_pos = np.arange(len(feature_stats))
    width = 0.35
    feature_names_range_iqr = [format_feature_name(idx, feature_mapping, 8) for idx in feature_stats['feature_idx']]
    plt.bar(x_pos - width/2, feature_stats['range_mean'], width, label='Range', alpha=0.8)
    plt.bar(x_pos + width/2, feature_stats['iqr_mean'], width, label='IQR', alpha=0.8)
    plt.title(f'Range vs IQR by Feature (All {n_features} Features)')
    plt.xlabel('Feature')
    plt.ylabel('Average Value')
    step_size = max(1, len(feature_names_range_iqr)//15)
    plt.xticks(range(0, len(feature_names_range_iqr), step_size), 
               [feature_names_range_iqr[i] for i in range(0, len(feature_names_range_iqr), step_size)], 
               rotation=45, ha='right')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_features_comprehensive_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. DETAILED BOXPLOTS (CHUNKED FOR READABILITY BUT COVERING ALL FEATURES)
    chunk_size = 10
    n_chunks = (n_features + chunk_size - 1) // chunk_size
    
    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, n_features)
        chunk_features = unique_features[start_idx:end_idx]
        
        if len(chunk_features) > 0:
            chunk_data = window_sample[window_sample['feature_idx'].isin(chunk_features)]
            
            # Get feature names for this chunk
            chunk_feature_names = [format_feature_name(idx, feature_mapping, 12) for idx in chunk_features]
            
            plt.figure(figsize=(16, 12))
            
            plt.subplot(2, 2, 1)
            sns.boxplot(data=chunk_data, x='feature_idx', y='mean')
            plt.title(f'Mean Distribution (Features {start_idx}-{end_idx-1})')
            plt.xlabel('Feature')
            plt.ylabel('Mean')
            plt.xticks(range(len(chunk_features)), chunk_feature_names, rotation=45, ha='right')
            
            plt.subplot(2, 2, 2)
            sns.boxplot(data=chunk_data, x='feature_idx', y='std')
            plt.title(f'Std Distribution (Features {start_idx}-{end_idx-1})')
            plt.xlabel('Feature')
            plt.ylabel('Standard Deviation')
            plt.xticks(range(len(chunk_features)), chunk_feature_names, rotation=45, ha='right')
            
            plt.subplot(2, 2, 3)
            sns.boxplot(data=chunk_data, x='feature_idx', y='skewness')
            plt.title(f'Skewness Distribution (Features {start_idx}-{end_idx-1})')
            plt.xlabel('Feature')
            plt.ylabel('Skewness')
            plt.xticks(range(len(chunk_features)), chunk_feature_names, rotation=45, ha='right')
            
            plt.subplot(2, 2, 4)
            sns.boxplot(data=chunk_data, x='feature_idx', y='range')
            plt.title(f'Range Distribution (Features {start_idx}-{end_idx-1})')
            plt.xlabel('Feature')
            plt.ylabel('Range')
            plt.xticks(range(len(chunk_features)), chunk_feature_names, rotation=45, ha='right')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'features_boxplot_chunk_{chunk_idx+1}_with_names.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    # 4. COMPREHENSIVE CORRELATION ANALYSIS (ALL FEATURES)
    print("Creating comprehensive correlation analysis for all features...")
    
    pivot_data = window_sample.pivot_table(values='mean', index='window_id', columns='feature_idx', aggfunc='first')
    
    if len(pivot_data.columns) > 1:
        corr_matrix = pivot_data.corr()
        
        # Create feature name labels for correlation matrix
        feature_labels = [format_feature_name(idx, feature_mapping, 10) for idx in corr_matrix.index]
        
        # Full correlation heatmap with names (ALL FEATURES)
        plt.figure(figsize=(max(16, n_features), max(14, n_features)))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Show only lower triangle
        
        # Create a copy of the correlation matrix with feature names as index/columns
        corr_matrix_named = corr_matrix.copy()
        corr_matrix_named.index = feature_labels
        corr_matrix_named.columns = feature_labels
        
        sns.heatmap(corr_matrix_named, mask=mask, annot=False, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title(f'Feature Correlation Matrix (All {len(corr_matrix)} Features)')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_correlation_full_with_names.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Correlation distribution analysis
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        corr_values = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
        plt.hist(corr_values, bins=50, alpha=0.7, edgecolor='black')
        plt.title('Distribution of All Feature Correlations')
        plt.xlabel('Correlation Coefficient')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        high_corr = corr_values[np.abs(corr_values) > 0.7]
        plt.hist(high_corr, bins=30, alpha=0.7, edgecolor='black')
        plt.title('High Correlations (|r| > 0.7)')
        plt.xlabel('Correlation Coefficient')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 3)
        # Feature with highest average absolute correlation
        avg_abs_corr = np.abs(corr_matrix).mean().sort_values(ascending=False)
        top_10_corr = avg_abs_corr.head(10)
        feature_names_top = [format_feature_name(idx, feature_mapping, 15) for idx in top_10_corr.index]
        plt.barh(range(len(top_10_corr)), top_10_corr.values)
        plt.title('Top 10 Features by Avg Absolute Correlation')
        plt.xlabel('Average |Correlation|')
        plt.yticks(range(len(feature_names_top)), feature_names_top)
        
        plt.subplot(2, 2, 4)
        # Feature with lowest average absolute correlation (most unique)
        bottom_10_corr = avg_abs_corr.tail(10)
        feature_names_bottom = [format_feature_name(idx, feature_mapping, 15) for idx in bottom_10_corr.index]
        plt.barh(range(len(bottom_10_corr)), bottom_10_corr.values)
        plt.title('Top 10 Most Unique Features (Low Correlation)')
        plt.xlabel('Average |Correlation|')
        plt.yticks(range(len(feature_names_bottom)), feature_names_bottom)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'correlation_analysis_comprehensive.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Highly correlated features with names
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:  # High correlation threshold
                    feature1_name = format_feature_name(corr_matrix.columns[i], feature_mapping)
                    feature2_name = format_feature_name(corr_matrix.columns[j], feature_mapping)
                    corr_pairs.append((corr_matrix.columns[i], feature1_name, 
                                     corr_matrix.columns[j], feature2_name, corr_val))
        
        if corr_pairs:
            print(f"Found {len(corr_pairs)} highly correlated feature pairs (|r| > 0.7)")
            # Save to file with names
            corr_df = pd.DataFrame(corr_pairs, columns=['Feature1_Idx', 'Feature1_Name', 
                                                       'Feature2_Idx', 'Feature2_Name', 'Correlation'])
            corr_df.to_csv(os.path.join(output_dir, 'high_correlations_with_names.csv'), index=False)
        else:
            print("No highly correlated feature pairs found (|r| > 0.7)")
    
    # 5. STATISTICS BY LABEL (ALL FEATURES)
    plt.figure(figsize=(20, 12))
    unique_labels = window_sample['label'].unique()
    
    for i, stat in enumerate(['mean', 'std', 'skewness', 'range']):
        plt.subplot(2, 2, i + 1)
        for label in unique_labels:  # All labels
            label_data = window_sample[window_sample['label'] == label]
            if len(label_data) > 0:
                feature_stats_by_label = label_data.groupby('feature_idx')[stat].mean()
                plt.plot(feature_stats_by_label.index, feature_stats_by_label.values, 
                        marker='o', label=f'Label {label}', alpha=0.7, markersize=3)
        
        plt.title(f'Average {stat.capitalize()} by Feature and Label (All Features)')
        plt.xlabel('Feature')
        plt.ylabel(f'Average {stat.capitalize()}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        # Create feature name labels for x-axis
        feature_indices = range(0, n_features, max(1, n_features//10))
        feature_names_labels = [format_feature_name(idx, feature_mapping, 8) for idx in feature_indices]
        plt.xticks(feature_indices, feature_names_labels, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'stats_by_label_all_features.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. SAVE COMPREHENSIVE FEATURE STATISTICS WITH NAMES
    feature_summary = create_feature_summary_with_names(window_sample, feature_mapping)
    feature_summary.to_csv(os.path.join(output_dir, 'feature_statistics_summary_with_names.csv'), index=False)
    
    # Save feature ranking by different criteria
    feature_ranking = pd.DataFrame({
        'feature_idx': feature_stats['feature_idx'],
        'feature_name': [format_feature_name(idx, feature_mapping) for idx in feature_stats['feature_idx']],
        'avg_mean': feature_stats['mean_mean'],
        'variability_of_means': feature_stats['mean_std'],
        'avg_std': feature_stats['std_mean'],
        'avg_skewness': np.abs(feature_stats['skewness_mean']),
        'avg_range': feature_stats['range_mean']
    })
    
    # Sort by different criteria and save
    for criterion in ['variability_of_means', 'avg_std', 'avg_skewness', 'avg_range']:
        ranked = feature_ranking.sort_values(criterion, ascending=False)
        ranked.to_csv(os.path.join(output_dir, f'feature_ranking_by_{criterion}.csv'), index=False)
    
    print(f"Window visualizations saved to {output_dir}/ with comprehensive analysis of all {n_features} features")

def visualize_subject_statistics(subject_stats_df, output_dir="data_exploration", config=None):
    """Create visualizations for subject-level statistics with feature names"""
    print("\n--- Creating Subject-Level Visualizations ---")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get feature mapping
    feature_mapping = get_feature_names_mapping(config) if config else {}
    
    unique_features = sorted(subject_stats_df['feature_idx'].unique())
    unique_subjects = sorted(subject_stats_df['subject_id'].unique())
    n_features = len(unique_features)
    n_subjects = len(unique_subjects)
    
    print(f"Creating subject visualizations for {n_subjects} subjects and {n_features} features...")
    
    # 1. OVERVIEW PLOTS WITH FEATURE NAMES
    plt.figure(figsize=(20, 15))
    
    # Subject variability across ALL features
    plt.subplot(2, 3, 1)
    feature_variability = subject_stats_df.groupby('feature_idx')['overall_mean'].std().reset_index()
    feature_names = [format_feature_name(idx, feature_mapping, 12) for idx in feature_variability['feature_idx']]
    plt.bar(range(len(feature_variability)), feature_variability['overall_mean'])
    plt.title('Between-Subject Variability by Feature (All Features)')
    plt.xlabel('Feature')
    plt.ylabel('Std of Subject Means')
    # Show feature names on x-axis, with intelligent spacing
    step = max(1, len(feature_names)//10)
    plt.xticks(range(0, len(feature_names), step), 
               [feature_names[i] for i in range(0, len(feature_names), step)], 
               rotation=45, ha='right')
    
    # Number of windows per subject
    plt.subplot(2, 3, 2)
    windows_per_subject = subject_stats_df.groupby('subject_id')['n_windows'].first()
    plt.bar(range(len(windows_per_subject)), windows_per_subject.values)
    plt.title('Number of Windows per Subject')
    plt.xlabel('Subject')
    plt.ylabel('Number of Windows')
    subject_labels = [str(s)[-3:] if len(str(s)) > 10 else str(s) for s in windows_per_subject.index]  # Truncate long subject names
    plt.xticks(range(len(subject_labels)), subject_labels, rotation=45)
    
    # Within vs Between subject variability (colored by feature)
    plt.subplot(2, 3, 3)
    scatter = plt.scatter(subject_stats_df['mean_of_means'], subject_stats_df['std_of_means'], 
                         alpha=0.6, s=20, c=subject_stats_df['feature_idx'], cmap='tab20')
    plt.xlabel('Mean of Window Means')
    plt.ylabel('Std of Window Means')
    plt.title('Within-Subject Variability (colored by feature)')
    cbar = plt.colorbar(scatter, label='Feature Index')
    # Add some feature name examples to the colorbar if there aren't too many features
    if n_features <= 20:
        tick_positions = np.linspace(0, n_features-1, min(5, n_features))
        tick_labels = [format_feature_name(int(pos), feature_mapping, 8) for pos in tick_positions]
        cbar.set_ticks(tick_positions)
        cbar.set_ticklabels(tick_labels)
    
    # Subject statistics distribution
    plt.subplot(2, 3, 4)
    sns.histplot(data=subject_stats_df, x='overall_std', bins=50)
    plt.title('Distribution of Subject Overall Std')
    
    # Feature ranking by variability (ALL features) with names
    plt.subplot(2, 3, 5)
    feature_var_ranking = subject_stats_df.groupby('feature_idx')['overall_std'].mean().sort_values(ascending=False)
    top_10_features = feature_var_ranking.head(10)  # Show top 10 most variable
    feature_names_top = [format_feature_name(idx, feature_mapping, 15) for idx in top_10_features.index]
    plt.barh(range(len(top_10_features)), top_10_features.values)
    plt.title('Top 10 Most Variable Features')
    plt.xlabel('Average Standard Deviation')
    plt.ylabel('Feature')
    plt.yticks(range(len(feature_names_top)), feature_names_top)
    
    # Subject consistency across features
    plt.subplot(2, 3, 6)
    subject_consistency = subject_stats_df.groupby('subject_id')['std_of_means'].mean()
    plt.bar(range(len(subject_consistency)), subject_consistency.values)
    plt.title('Subject Consistency (Lower = More Consistent)')
    plt.xlabel('Subject')
    plt.ylabel('Average Std of Window Means')
    subject_labels = [str(s)[-3:] if len(str(s)) > 10 else str(s) for s in subject_consistency.index]
    plt.xticks(range(len(subject_labels)), subject_labels, rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'subject_statistics_overview_with_names.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. SUBJECT-FEATURE HEATMAPS WITH FEATURE NAMES
    print("Creating subject-feature heatmaps with feature names...")
    
    # Create heatmap of subject means across ALL features
    subject_feature_means = subject_stats_df.pivot(index='subject_id', columns='feature_idx', values='overall_mean')
    
    # Create feature name labels
    feature_labels = [format_feature_name(idx, feature_mapping, 10) for idx in subject_feature_means.columns]
    subject_labels = [str(s)[-6:] if len(str(s)) > 10 else str(s) for s in subject_feature_means.index]  # Truncate long subject names
    
    plt.figure(figsize=(max(20, n_features), max(8, n_subjects)))
    
    # Create a copy with proper labels
    heatmap_data = subject_feature_means.copy()
    heatmap_data.columns = feature_labels
    heatmap_data.index = subject_labels
    
    sns.heatmap(heatmap_data, annot=False, cmap='viridis', cbar_kws={"shrink": 0.8})
    plt.title(f'Subject-Feature Mean Values Heatmap ({n_subjects} Subjects × {n_features} Features)')
    plt.xlabel('Feature')
    plt.ylabel('Subject')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'subject_feature_heatmap_means_with_names.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create heatmap of subject variability across ALL features
    subject_feature_stds = subject_stats_df.pivot(index='subject_id', columns='feature_idx', values='overall_std')
    
    plt.figure(figsize=(max(20, n_features), max(8, n_subjects)))
    
    # Create a copy with proper labels
    heatmap_data_std = subject_feature_stds.copy()
    heatmap_data_std.columns = feature_labels
    heatmap_data_std.index = subject_labels
    
    sns.heatmap(heatmap_data_std, annot=False, cmap='plasma', cbar_kws={"shrink": 0.8})
    plt.title(f'Subject-Feature Variability Heatmap ({n_subjects} Subjects × {n_features} Features)')
    plt.xlabel('Feature')
    plt.ylabel('Subject')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'subject_feature_heatmap_stds_with_names.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. FEATURE ANALYSIS SUMMARIES WITH NAMES
    # Most/least variable features with names
    feature_analysis = subject_stats_df.groupby('feature_idx').agg({
        'overall_mean': ['mean', 'std', 'min', 'max'],
        'overall_std': ['mean', 'std'],
        'mean_of_means': ['mean', 'std'],
        'std_of_means': ['mean', 'std']
    }).round(4)
    
    feature_analysis.columns = ['_'.join(col).strip() for col in feature_analysis.columns]
    feature_analysis = feature_analysis.reset_index()
    
    # Add feature names
    feature_analysis['feature_name'] = feature_analysis['feature_idx'].map(
        lambda x: feature_mapping.get(x, f"Feature_{x}")
    )
    
    # Reorder columns
    cols = ['feature_idx', 'feature_name'] + [col for col in feature_analysis.columns if col not in ['feature_idx', 'feature_name']]
    feature_analysis = feature_analysis[cols]
    feature_analysis.to_csv(os.path.join(output_dir, 'feature_analysis_complete_with_names.csv'), index=False)
    
    # Subject analysis summary
    subject_analysis = subject_stats_df.groupby('subject_id').agg({
        'overall_mean': ['mean', 'std'],
        'overall_std': ['mean', 'std'],
        'n_windows': 'first',
        'mean_of_means': ['mean', 'std'],
        'std_of_means': ['mean', 'std']
    }).round(4)
    
    subject_analysis.columns = ['_'.join(col).strip() for col in subject_analysis.columns]
    subject_analysis = subject_analysis.reset_index()
    subject_analysis.to_csv(os.path.join(output_dir, 'subject_analysis_complete.csv'), index=False)
    
    # Create feature ranking summary
    feature_ranking = subject_stats_df.groupby('feature_idx').agg({
        'overall_std': 'mean',
        'std_of_means': 'mean'
    }).round(4)
    feature_ranking['feature_name'] = feature_ranking.index.map(
        lambda x: feature_mapping.get(x, f"Feature_{x}")
    )
    feature_ranking = feature_ranking.reset_index()
    feature_ranking = feature_ranking.sort_values('overall_std', ascending=False)
    feature_ranking.to_csv(os.path.join(output_dir, 'feature_variability_ranking_with_names.csv'), index=False)
    
    print(f"Subject visualizations saved to {output_dir}/ with feature names")
    print(f"Saved comprehensive analysis for all {n_subjects} subjects and {n_features} features")

def save_and_load_parallel_results(output_dir, window_df=None, subject_df=None, load_only=False):
    """Save or load parallel processing results"""
    os.makedirs(output_dir, exist_ok=True)
    
    window_stats_path = os.path.join(output_dir, "window_statistics_parallel.pkl")
    subject_stats_path = os.path.join(output_dir, "subject_statistics_parallel.pkl")
    
    if load_only:
        # Try to load existing results
        window_df = None
        subject_df = None
        
        if os.path.exists(window_stats_path):
            print("Loading existing window statistics...")
            with open(window_stats_path, 'rb') as f:
                window_df = pickle.load(f)
        
        if os.path.exists(subject_stats_path):
            print("Loading existing subject statistics...")
            with open(subject_stats_path, 'rb') as f:
                subject_df = pickle.load(f)
        
        return window_df, subject_df
    
    else:
        # Save results
        if window_df is not None:
            print("Saving window statistics...")
            with open(window_stats_path, 'wb') as f:
                pickle.dump(window_df, f)
            # Also save as CSV
            window_df.to_csv(os.path.join(output_dir, "window_statistics.csv"), index=False)
        
        if subject_df is not None:
            print("Saving subject statistics...")
            with open(subject_stats_path, 'wb') as f:
                pickle.dump(subject_df, f)
            # Also save as CSV
            subject_df.to_csv(os.path.join(output_dir, "subject_statistics.csv"), index=False)

def generate_summary_report(window_stats_df, subject_stats_df, X_windows=None, y_windows=None, subject_ids_windows=None, config=None):
    """Generate a comprehensive summary report"""
    print("\n" + "="*50)
    print("DATA EXPLORATION SUMMARY REPORT (PARALLEL)")
    print("="*50)
    
    # Get feature mapping for display
    feature_mapping = get_feature_names_mapping(config) if config else {}
    
    # Basic data info
    print(f"\nDATASET OVERVIEW:")
    if X_windows is not None:
        print(f"  Total windows: {len(X_windows):,}")
        print(f"  Window length: {X_windows.shape[1]} timesteps")
        print(f"  Number of features: {X_windows.shape[2]}")
        print(f"  Number of subjects: {len(np.unique(subject_ids_windows))}")
        print(f"  Number of unique labels: {len(np.unique(y_windows))}")
    else:
        print(f"  Total windows analyzed: {len(window_stats_df):,}")
        print(f"  Total subjects: {subject_stats_df['subject_id'].nunique():,}")
        print(f"  Total features: {window_stats_df['feature_idx'].nunique():,}")
        print(f"  Unique labels: {sorted(window_stats_df['label'].unique())}")
    
    # Label distribution
    if y_windows is not None:
        unique_labels, label_counts = np.unique(y_windows, return_counts=True)
        print(f"\nLABEL DISTRIBUTION:")
        for label, count in zip(unique_labels, label_counts):
            percentage = (count / len(y_windows)) * 100
            print(f"  Label {label}: {count:,} windows ({percentage:.1f}%)")
    else:
        print(f"\nLABEL DISTRIBUTION:")
        label_dist = window_stats_df['label'].value_counts().sort_index()
        for label, count in label_dist.items():
            percentage = (count / len(window_stats_df)) * 100
            print(f"  Label {label}: {count:,} windows ({percentage:.1f}%)")
    
    # Subject distribution
    print(f"\nSUBJECT DISTRIBUTION:")
    if subject_ids_windows is not None:
        subject_window_counts = pd.Series(subject_ids_windows).value_counts().sort_index()
    else:
        subject_window_counts = window_stats_df.groupby('subject_id').size()
    
    print(f"  Average windows per subject: {subject_window_counts.mean():.1f}")
    print(f"  Min windows per subject: {subject_window_counts.min()}")
    print(f"  Max windows per subject: {subject_window_counts.max()}")
    
    # Feature statistics summary
    print(f"\nFEATURE STATISTICS SUMMARY:")
    
    print("  Top 5 most variable features (by std of means):")
    std_of_means = window_stats_df.groupby('feature_idx')['mean'].std().sort_values(ascending=False)
    for i, (feature_idx, std_val) in enumerate(std_of_means.head().items()):
        feature_name = format_feature_name(feature_idx, feature_mapping)
        print(f"    {feature_name} (idx {feature_idx}): std = {std_val:.4f}")
    
    print("  Top 5 most skewed features (by mean absolute skewness):")
    mean_abs_skew = window_stats_df.groupby('feature_idx')['skewness'].apply(lambda x: np.mean(np.abs(x))).sort_values(ascending=False)
    for i, (feature_idx, skew_val) in enumerate(mean_abs_skew.head().items()):
        feature_name = format_feature_name(feature_idx, feature_mapping)
        print(f"    {feature_name} (idx {feature_idx}): mean |skewness| = {skew_val:.4f}")
    
    # Subject variability
    print(f"\nSUBJECT VARIABILITY:")
    subject_variability = subject_stats_df.groupby('subject_id')['std_of_means'].mean().sort_values(ascending=False)
    print(f"  Most variable subject: {subject_variability.index[0]} (avg std of means: {subject_variability.iloc[0]:.4f})")
    print(f"  Least variable subject: {subject_variability.index[-1]} (avg std of means: {subject_variability.iloc[-1]:.4f})")
    
    # Processing info
    print(f"\nPROCESSING INFO:")
    print(f"  Used parallel processing with multiprocessing")
    print(f"  Results cached for fast reloading")
    print(f"  Feature names mapped from config.yaml sensor_columns_original")
    
    print("\n" + "="*50)
    print("Exploration complete! Check the generated plots for detailed visualizations.")
    print("="*50)

def main():
    """Main function to run the parallel data exploration"""
    print("Starting Parallel Data Exploration...")
    print(f"Available CPUs: {cpu_count()}")
    
    # Load configuration
    config = load_config()
    project_root = os.getcwd()
    
    # Use conservative number of processes to avoid memory issues
    n_jobs = min(4, cpu_count())
    output_dir = "data_exploration"
    
    print(f"Using {n_jobs} processes for analysis")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if results already exist
    existing_window_df, existing_subject_df = save_and_load_parallel_results(output_dir, load_only=True)
    
    if existing_window_df is not None and existing_subject_df is not None:
        print("Found existing analysis results. Using cached data.")
        window_stats_df = existing_window_df
        subject_stats_df = existing_subject_df
        X_windows, y_windows, subject_ids_windows = None, None, None
    else:
        print("No existing results found. Starting fresh analysis...")
        
        # Load data from pipeline output files
        data_result = prepare_all_data_for_subject_optimization(config, project_root, [])
        if data_result is None:
            print("Failed to load data from pipeline output. Exiting.")
            print("Please run pipeline.ipynb first to create the required files:")
            print("  - /scai_data3/scratch/stirnimann_r/results/pipeline/combined_windowed_dataset_mapped_filtered.pkl")
            print("  - /scai_data3/scratch/stirnimann_r/results/pipeline/combined_dataset_metadata_mapped_filtered.json")
            return
        
        X_windows, y_windows, subject_ids_windows = data_result
        
        print(f"Loaded pipeline data - X_windows: {X_windows.shape}, y_windows: {y_windows.shape}")
        print(f"Unique subjects: {len(np.unique(subject_ids_windows))}")
        print(f"Memory usage optimized through pipeline preprocessing")
        
        # Parallel analysis
        print(f"\nStarting parallel analysis with {n_jobs} processes...")
        
        # Analyze window statistics
        window_stats_df = analyze_window_statistics_parallel(X_windows, y_windows, subject_ids_windows, n_jobs=n_jobs)
        
        # Analyze subject statistics
        subject_stats_df = analyze_subject_statistics_parallel(X_windows, y_windows, subject_ids_windows, n_jobs=n_jobs)
        
        # Save results
        save_and_load_parallel_results(output_dir, window_df=window_stats_df, subject_df=subject_stats_df)
    
    # Create visualizations with feature names
    print("\nCreating visualizations with feature names...")
    visualize_window_statistics(window_stats_df, output_dir, config)
    visualize_subject_statistics(subject_stats_df, output_dir, config)
    
    # Generate summary report
    print("\nGenerating summary report...")
    if X_windows is not None:
        generate_summary_report(window_stats_df, subject_stats_df, X_windows, y_windows, subject_ids_windows, config)
    else:
        generate_summary_report(window_stats_df, subject_stats_df, config=config)
    
    print(f"\nAll results saved to {output_dir}/:")
    print(f"  - window_statistics.csv")
    print(f"  - subject_statistics.csv")
    print(f"  - Various visualization PNG files")
    print(f"Analysis completed using {n_jobs} parallel processes with pipeline data.")

if __name__ == "__main__":
    main()