# Master_Thesis_Ricardo

This project implements a data processing and machine learning pipeline for sensor data.

## Prerequisites

Before running the pipeline, ensure the following directories and files are set up correctly:

1.  **Project Root Directory**:
    *   This is the main directory where you have cloned or set up the project.
    *   It should contain the `src` directory, `config.yaml`, `Sync_Parameters.yaml`, `All_Videos_with_Labels_Real_Time_Corrected_Labels.csv`, and `Activity_Mapping.csv` (or ensure paths in `config.yaml` point to their correct locations if they are elsewhere).

2.  **Source Code (`src/`)**:
    *   The `src/` directory must contain all the Python scripts for the pipeline stages:
        *   `raw_data_processor.py`
        *   `debug_labels_v2.ipynb`

3.  **Configuration Files**:
    *   **`config.yaml`**:
        *   Must be present in the project root (or its path provided via the `--config` argument).
        *   Verify all paths within this file are correct, especially:
            *   `raw_data_input_dir`: Points to the directory containing raw subject data.
            *   `sync_parameters_file`: Path to `Sync_Parameters.yaml`.
            *   `global_labels_file`: Path to the global labels CSV file.
            *   `activity_mapping_file`: Path to the activity mapping CSV file.
        *   Ensure `subjects_to_process` and `subjects_to_load` lists are correctly populated if you are not processing all available subjects.
        *   Review `raw_data_parsing_config` to ensure `data_columns`, `sample_rate`, and `file_format` match your raw data files.
    *   **`Sync_Parameters.yaml`**:
        *   Path should be correctly specified in `config.yaml`.
        *   Contains synchronization details for each subject and sensor.
    *   **`All_Videos_with_Labels_Real_Time_Corrected_Labels.csv`** (or the file specified in `global_labels_file`):
        *   Path should be correctly specified in `config.yaml`.
        *   This file provides the ground truth labels.
    *   **`Activity_Mapping.csv`** (or the file specified in `activity_mapping_file`):
        *   Path should be correctly specified in `config.yaml`.
        *   Used for mapping raw activity labels to a consistent set.

4.  **Raw Data Input Directory** (as specified by `raw_data_input_dir` in `config.yaml`):
    *   This directory must exist. The default in the provided `config.yaml` is `"/scai_data2/scai_datasets/interim/scai-outsense/"`.
    *   Inside this directory, there should be sub-folders for each subject listed in `config.yaml` (e.g., `OutSense-036/`, `OutSense-115/`, etc.).
    *   Each subject's folder must contain the raw sensor data files (e.g., `.csv.gz` files). The filenames and internal structure (column names) of these CSV files should correspond to what is defined in the `raw_data_parsing_config` section of `config.yaml` for each sensor type.

**Note**: The pipeline will create the following directories if they don't exist (based on `config.yaml` settings):
*   `results_dir` (default: `results/`)
*   `processed_data_output_dir` (default: `processed_subjects/`)
*   `intermediate_feature_dir` (default: `features/`)
Ensure that the parent directory of the project has write permissions for these to be created.

## Running the Pipeline

The pipeline should be executed by running the `raw_data_processor.py` script located in the `src` directory first and in a second step using `debug_labels_v2.ipynb` to get the final dataset used for training (execute all cells).

Then either the XGBoost model or the MCA-TCN models can be trained by executing the training files.

