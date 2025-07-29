# Master_Thesis_Ricardo

This project implements a data processing and machine learning pipeline for sensor data.

## Prerequisites

Before running the pipeline, ensure the following directories and files are set up correctly:

1.  **Project Root Directory**:
    *   This is the main directory where you have cloned or set up the project.
    *   It should contain the `src` directory, `config.yaml`, `Sync_Parameters.yaml`, `All_Videos_with_Labels_Real_Time_Corrected_Labels.csv`, and `Activity_Mapping.csv` (or ensure paths in `config.yaml` point to their correct locations if they are elsewhere).

2.  **Source Code (`src/`)**:
    *   The `src/` directory must contain all the Python scripts for the pipeline stages:
        *   `main_pipeline.py`
        *   `config_loader.py`
        *   `raw_data_processor.py`
        *   `data_loader.py`
        *   `feature_engineering.py`
        *   `feature_selector.py`
        *   `data_preparation.py`
        *   `training.py`
        *   `evaluation.py`
        *   `models.py`
        *   `utils.py`

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

The main pipeline can be executed by running the `main_pipeline.py` script located in the `src` directory.

```bash
python src/main_pipeline.py
```

### Command-Line Arguments

The `main_pipeline.py` script accepts several command-line arguments to customize its execution:

*   `--config <path_to_config>`: Specifies the path to the configuration YAML file. Defaults to `config.yaml` in the project root.
*   `--start-stage <stage_name>`: Defines the stage at which the pipeline should begin.
    *   Choices: `raw_data_processing`, `load_data`, `feature_engineering`, `feature_selection`, `data_preparation`, `training`, `evaluation`.
*   `--stop-stage <stage_name>`: Defines the stage after which the pipeline should stop.
    *   Choices: `raw_data_processing`, `load_data`, `feature_engineering`, `feature_selection`, `data_preparation`, `training`, `evaluation`.
*   `--log-level <level>`: Sets the logging level.
    *   Choices: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`. Defaults to `INFO`.
*   `--force-run <stage_name_1> <stage_name_2> ...`: Forces the specified stage(s) to run even if their outputs already exist.

### Examples

Here are some examples of how to run the pipeline from the terminal:

1.  **Run the entire pipeline with default settings:**
    ```bash
    python src/main_pipeline.py
    ```

2.  **Run only the `raw_data_processing` stage:**
    ```bash
    python src/main_pipeline.py --start-stage raw_data_processing --stop-stage raw_data_processing
    ```

3.  **Start the pipeline from the `feature_engineering` stage and run until the end:**
    ```bash
    python src/main_pipeline.py --start-stage feature_engineering
    ```

4.  **Run the pipeline up to the `data_preparation` stage:**
    ```bash
    python src/main_pipeline.py --stop-stage data_preparation
    ```

5.  **Force the `training` stage to re-run, even if a model file exists:**
    ```bash
    python src/main_pipeline.py --start-stage training --stop-stage training --force-run training
    ```

6.  **Use a custom configuration file named `my_custom_config.yaml`:**
    ```bash
    python src/main_pipeline.py --config my_custom_config.yaml
    ```

7.  **Run the pipeline with a `DEBUG` logging level:**
    ```bash
    python src/main_pipeline.py --log-level DEBUG
    ```

### Configuration

The pipeline's behavior is controlled by two main configuration files:

1.  **`config.yaml`**: This file contains general settings, paths for input/output directories, and parameters for each stage of the pipeline, including:
    *   Raw data processing
    *   Data loading and cleaning
    *   Feature engineering
    *   Feature selection
    *   Data preparation
    *   Model training
    *   Evaluation

    You can modify this file to change parameters such as the list of subjects to process, sensor-specific parsing configurations, filter parameters, model types, learning rates, etc.

2.  **`Sync_Parameters.yaml`**: This file defines synchronization parameters (time shifts and drift corrections) for different sensors and subjects. It is used during the raw data processing stage to align sensor readings.

### Pipeline Stages

The pipeline is divided into several stages, each handled by a corresponding Python script in the `src` directory:

*   **Stage 0: Raw Data Processing (`raw_data_processor.py`)**: Processes raw sensor data from CSV files. It handles tasks like resampling, filtering, and applying synchronization corrections based on `Sync_Parameters.yaml`.
*   **Stage 1: Data Loading & Cleaning (`data_loader.py`)**: Loads the processed data from Stage 0, performs cleaning, and maps activity labels.
*   **Stage 2: Feature Engineering (`feature_engineering.py`)**: Generates features from the cleaned data using specified windowing techniques.
*   **Stage 3: Feature Selection (`feature_selector.py`)**: Selects relevant features for model training.
*   **Stage 4: Data Preparation (`data_preparation.py`)**: Prepares the data for model training, including splitting into training and testing sets.
*   **Stage 5: Model Training (`training.py`)**: Trains the specified machine learning model using the prepared data.
*   **Stage 6: Evaluation (`evaluation.py`)**: Evaluates the trained model using various metrics.

To run specific parts of the pipeline or to customize its execution, you will need to adjust the settings in `config.yaml`. For example, to process a different set of subjects, update the `subjects_to_process` list under the `raw_data_processing` section or `subjects_to_load` under the `data_loading_cleaning` section. Similarly, model parameters, learning rates, and other hyperparameters can be tuned in this file.
