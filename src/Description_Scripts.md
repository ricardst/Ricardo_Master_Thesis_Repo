# Description of Scripts and Notebooks in src/

This document provides a brief description of all Python scripts and Jupyter notebooks in the `src/` folder of the Master Thesis project.

## Core Pipeline Scripts

### `raw_data_processor.py`
Processes raw sensor data files. Applies filtering (Butterworth filters), handles data synchronization, and prepares raw data for the data loading pipeline.

### `debug_labels_v2.ipynb`
Debugging notebook for label verification and correction, ensuring data quality. The output is needed by the training pipeline.

## Model and Training Scripts

### `pipeline.ipynb`
Interactive notebook demonstrating the complete machine learning pipeline with step-by-step execution and visualizations.

### `XGBoost_Pipline.ipynb`
Implementation and evaluation of XGBoost models for the activity classification task.

### `tcn_activity_classification.py`
Standalone script for training and evaluating Temporal Convolutional Network (TCN) models specifically for activity classification tasks.

### `tcn_activity_classification_CVtuning.py`
Extended version of TCN activity classification with cross-validation and hyperparameter tuning capabilities.

### `mca_tcn_activity_classification_CVtuning.py`
TCN activity classification with Multiple Correspondence Analysis (MCA) integration and cross-validation tuning.

### `train_ae.py`
Training script specifically for autoencoder models used in unsupervised learning and feature extraction.

## Data Analysis and Segmentation

### `ae_segmentation.py`
Uses autoencoders for time series segmentation and change point detection in sensor data streams.

### `claspy_segmentation.py`
Implements CLASP (CLuster Aware Seasonal Patterns) algorithm for time series segmentation and pattern detection.

### `data_analyzer.py`
Provides data analysis utilities and statistical analysis of the sensor datasets.

### `plot_feature_data.py`
Visualization utilities for plotting feature data, sensor signals, and analysis results.

### `quick_data_analysis.py`
Quick analysis scripts for rapid data exploration and validation.

## Hyperparameter Optimization

### `HPO_tracking.py`
Tracks and logs hyperparameter optimization experiments and results.

## Utility Scripts

### `config_loader.py`
Loads and manages configuration parameters from YAML files for consistent experiment setup.

### `utils.py`
Contains utility functions, data structures, and helper classes used across the project including dataset classes and common operations.

## Jupyter Notebooks

### `check_combined_data.ipynb`
Notebook for validating and exploring the combined dataset after data loading and preprocessing.

### `debug_labels_v2.ipynb`
Debugging notebook for label verification and correction, ensuring data quality.

### `debug_raw_processing.ipynb`
Debugging notebook for raw data processing validation and troubleshooting.

## Summary

The src/ folder contains a comprehensive machine learning pipeline for sensor-based activity recognition, including:
- Data processing and feature engineering pipelines
- Multiple neural network architectures (CNN, TCN, Autoencoders)
- Hyperparameter optimization frameworks
- Evaluation and visualization tools
- Interactive notebooks for experimentation and debugging

All scripts are designed to work together as part of a coordinated pipeline while also being capable of standalone execution for specific tasks.