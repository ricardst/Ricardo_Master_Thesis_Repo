#!/usr/bin/env python3
"""
Quick Pipeline Data Analysis
===========================

A simplified analysis script that works with the windowed dataset output
from pipeline.ipynb. Provides quick insights into the processed windowed data,
activity distribution, and sensor coverage.

Usage:
    python quick_data_analysis.py
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pickle
import json

# Add src to path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# ...existing code...


def load_pipeline_data():
    """Load the combined windowed dataset from pipeline output"""
    
    # Pipeline output paths
    pipeline_dir = os.path.join(script_dir, '..', 'results', 'pipeline')
    dataset_path = os.path.join(pipeline_dir, 'combined_windowed_dataset_mapped_filtered.pkl')
    metadata_path = os.path.join(pipeline_dir, 'combined_dataset_metadata_mapped_filtered.json')
    
    # Try alternative paths if files don't exist
    alt_dataset_path = os.path.join(pipeline_dir, 'combined_windowed_dataset.pkl')
    alt_metadata_path = os.path.join(pipeline_dir, 'combined_dataset_metadata.json')
    
    print("Loading pipeline data...")
    
    # Load dataset
    if os.path.exists(dataset_path):
        print(f"  Loading: {dataset_path}")
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
    elif os.path.exists(alt_dataset_path):
        print(f"  Loading: {alt_dataset_path}")
        with open(alt_dataset_path, 'rb') as f:
            dataset = pickle.load(f)
    else:
        raise FileNotFoundError(f"No dataset found at {dataset_path} or {alt_dataset_path}")
    
    # Load metadata
    metadata = {}
    if os.path.exists(metadata_path):
        print(f"  Loading: {metadata_path}")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    elif os.path.exists(alt_metadata_path):
        print(f"  Loading: {alt_metadata_path}")
        with open(alt_metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        print("  Warning: No metadata file found")
    
    return dataset, metadata


def analyze_pipeline_data():
    """Analyze the windowed dataset from pipeline output"""
    
    try:
        print("Loading pipeline windowed dataset...")
        dataset, metadata = load_pipeline_data()
        
        # Extract data components
        windows = dataset['windows']  # Shape: (n_windows, window_size, n_features)
        labels = dataset['labels']    # Shape: (n_windows,)
        window_info = dataset['window_info']
        
        print(f"Pipeline data loaded successfully!")
        print(f"Windows shape: {windows.shape}")
        print(f"Labels: {len(labels):,}")
        print(f"Window info shape: {window_info.shape}")
        
        # Get sensor configuration
        sensor_config = dataset.get('sensor_config', {})
        sensor_columns = sensor_config.get('final_sensor_columns', [])
        
        print(f"Number of sensor features: {len(sensor_columns)}")
        
    except Exception as e:
        print(f"Error loading pipeline data: {e}")
        return
    
    # Check for subject information
    subject_column = 'SubjectID'
    if subject_column in window_info.columns:
        unique_subjects = window_info[subject_column].unique()
        print(f"Number of unique subjects: {len(unique_subjects)}")
        print(f"Subjects: {list(unique_subjects)}")
    else:
        print("Subject information not found in window_info")
        unique_subjects = []
    
    # Activity analysis
    unique_activities = np.unique(labels)
    print(f"Number of unique activities: {len(unique_activities)}")
    print(f"Activities: {list(unique_activities)}")
    
    # Quick class distribution analysis
    print("\n" + "="*50)
    print("ACTIVITY DISTRIBUTION ANALYSIS")
    print("="*50)
    
    activity_counts = Counter(labels)
    total_windows = len(labels)
    
    print(f"Total windows: {total_windows:,}")
    print(f"Unique activities: {len(activity_counts)}")
    
    for activity, count in sorted(activity_counts.items()):
        percentage = (count / total_windows) * 100
        print(f"{activity:<20}: {count:>6} windows ({percentage:>5.1f}%)")
    
    # Sensor analysis
    print("\n" + "="*50)
    print("SENSOR FEATURE ANALYSIS")
    print("="*50)
    
    print(f"Total sensor features: {windows.shape[2]}")
    print(f"Window size (time steps): {windows.shape[1]}")
    
    # Analyze sensor data quality
    print("\nSensor Data Quality:")
    
    # Check for missing values
    nan_count = np.isnan(windows).sum()
    total_values = windows.size
    nan_percentage = (nan_count / total_values) * 100
    print(f"  NaN values: {nan_count:,} ({nan_percentage:.3f}%)")
    
    # Check for infinite values
    inf_count = np.isinf(windows).sum()
    inf_percentage = (inf_count / total_values) * 100
    print(f"  Infinite values: {inf_count:,} ({inf_percentage:.3f}%)")
    
    # Feature statistics
    feature_means = np.nanmean(windows, axis=(0, 1))
    feature_stds = np.nanstd(windows, axis=(0, 1))
    
    print(f"  Feature value ranges:")
    print(f"    Mean: [{np.nanmin(feature_means):.3f}, {np.nanmax(feature_means):.3f}]")
    print(f"    Std:  [{np.nanmin(feature_stds):.3f}, {np.nanmax(feature_stds):.3f}]")
    
    # Display sensor column information if available
    if sensor_columns:
        print(f"\nSensor Column Examples:")
        for i, col in enumerate(sensor_columns[:10]):
            print(f"  {i+1:2d}. {col}")
        if len(sensor_columns) > 10:
            print(f"  ... and {len(sensor_columns) - 10} more sensors")
    
    # Subject-wise analysis
    if len(unique_subjects) > 0:
        print("\n" + "="*50)
        print("SUBJECT-WISE ANALYSIS")
        print("="*50)
        
        subject_counts = window_info[subject_column].value_counts()
        print(f"Windows per subject:")
        for subject, count in subject_counts.items():
            percentage = (count / total_windows) * 100
            print(f"  {subject}: {count:,} windows ({percentage:.1f}%)")
    
    # Create visualizations
    print("\n" + "="*50)
    print("CREATING VISUALIZATIONS")
    print("="*50)
    
    # Set up the plot
    plt.style.use('default')
    if len(unique_subjects) > 0:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    else:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Activity distribution
    activities = list(activity_counts.keys())
    counts = list(activity_counts.values())
    
    # Limit to top 15 activities for readability
    if len(activities) > 15:
        top_15_items = sorted(activity_counts.items(), key=lambda x: x[1], reverse=True)[:15]
        activities = [item[0] for item in top_15_items]
        counts = [item[1] for item in top_15_items]
    
    bars1 = ax1.bar(range(len(activities)), counts, color='skyblue')
    ax1.set_title('Activity Distribution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Activity')
    ax1.set_ylabel('Number of Windows')
    ax1.set_xticks(range(len(activities)))
    ax1.set_xticklabels(activities, rotation=45, ha='right')
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=8)
    
    # 2. Subject distribution (if available)
    if len(unique_subjects) > 0:
        subjects = list(subject_counts.index)
        subj_counts = list(subject_counts.values)
        
        bars2 = ax2.bar(range(len(subjects)), subj_counts, color='lightgreen')
        ax2.set_title('Windows per Subject', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Subject')
        ax2.set_ylabel('Number of Windows')
        ax2.set_xticks(range(len(subjects)))
        ax2.set_xticklabels(subjects, rotation=45, ha='right')
        
        # Add value labels
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=8)
    else:
        # Alternative plot: Window size distribution
        ax2.hist(np.random.choice(windows.flatten(), 10000), bins=50, color='lightgreen', alpha=0.7)
        ax2.set_title('Sample Feature Value Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Feature Value')
        ax2.set_ylabel('Frequency')
        ax2.set_yscale('log')
    
    # 3. Feature statistics
    if len(sensor_columns) > 0:
        # Show feature means for first 20 features
        n_features_to_show = min(20, len(feature_means))
        feature_indices = range(n_features_to_show)
        
        bars3 = ax3.bar(feature_indices, feature_means[:n_features_to_show], color='lightcoral')
        ax3.set_title(f'Feature Means (First {n_features_to_show})', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Feature Index')
        ax3.set_ylabel('Mean Value')
        ax3.tick_params(axis='x', rotation=45)
    else:
        ax3.text(0.5, 0.5, 'No sensor column\ninformation available', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Sensor Information', fontsize=14, fontweight='bold')
    
    # 4. Window metadata analysis (if available)
    if len(unique_subjects) > 0 and 'label_coverage' in window_info.columns:
        coverage_values = window_info['label_coverage']
        ax4.hist(coverage_values, bins=20, color='gold', alpha=0.7, edgecolor='black')
        ax4.set_title('Window Label Coverage Distribution', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Label Coverage')
        ax4.set_ylabel('Number of Windows')
        ax4.axvline(x=0.8, color='red', linestyle='--', label='80% threshold')
        ax4.legend()
    elif len(unique_subjects) > 0:
        # Alternative: Show activity distribution by subject
        subject_activity_data = []
        for subject in unique_subjects[:5]:  # Show first 5 subjects
            subject_mask = window_info[subject_column] == subject
            subject_labels = labels[subject_mask]
            subject_activity_counts = Counter(subject_labels)
            subject_activity_data.append([subject_activity_counts.get(act, 0) for act in activities[:5]])
        
        if subject_activity_data:
            subject_activity_matrix = np.array(subject_activity_data)
            im = ax4.imshow(subject_activity_matrix, cmap='YlOrRd', aspect='auto')
            ax4.set_title('Activity Distribution by Subject', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Activity')
            ax4.set_ylabel('Subject')
            ax4.set_xticks(range(len(activities[:5])))
            ax4.set_xticklabels([act[:10] for act in activities[:5]], rotation=45, ha='right')
            ax4.set_yticks(range(len(unique_subjects[:5])))
            ax4.set_yticklabels(unique_subjects[:5])
            plt.colorbar(im, ax=ax4, label='Number of Windows')
    
    plt.tight_layout()
    
    # Save the plot
    output_path = 'quick_pipeline_data_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    plt.show()
    
    # Print metadata information
    if metadata:
        print("\n" + "="*50)
        print("PIPELINE METADATA")
        print("="*50)
        
        print(f"Creation time: {metadata.get('creation_time', 'Unknown')}")
        print(f"Subjects included: {metadata.get('subjects_included', [])}")
        print(f"Total windows: {metadata.get('n_windows', 'Unknown'):,}")
        print(f"Total features: {metadata.get('n_features', 'Unknown')}")
        print(f"Unique labels: {metadata.get('n_unique_labels', 'Unknown')}")
        
        if 'activities_of_interest' in metadata:
            print(f"Activities of interest: {metadata['activities_of_interest']}")
        
        if 'activity_mapping_applied' in metadata:
            print(f"Activity mapping applied: {metadata['activity_mapping_applied']}")
    
    # Summary statistics
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    
    print(f"Total windows analyzed: {len(labels):,}")
    print(f"Total features per window: {windows.shape[2]}")
    print(f"Window size (time steps): {windows.shape[1]}")
    print(f"Unique activities: {len(unique_activities)}")
    if len(unique_subjects) > 0:
        print(f"Subjects: {len(unique_subjects)}")
    
    # Data quality overview
    completeness = (1 - nan_count / total_values) * 100
    print(f"Data completeness: {completeness:.1f}%")
    
    if inf_count == 0 and nan_count == 0:
        print("✅ No data quality issues detected")
    else:
        print("⚠️ Data quality issues detected - see details above")
    
    print("\nTop 5 most frequent activities:")
    for i, (activity, count) in enumerate(sorted(activity_counts.items(), key=lambda x: x[1], reverse=True)[:5]):
        percentage = (count / total_windows) * 100
        print(f"  {i+1}. {activity}: {count:,} windows ({percentage:.1f}%)")
    
    print("\n" + "="*50)
    print("ANALYSIS COMPLETE")
    print("="*50)
    
    return dataset, metadata

# ...existing code...


if __name__ == '__main__':
    analyze_pipeline_data()
