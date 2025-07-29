import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os

# --- Configuration ---
# Adjust these if your column names are different
SUBJECT_COLUMN = 'SubjectID'  # Column name for subject identifier
# TIMESTAMP_COLUMN = None # No longer a fixed global, will be chosen by user
# Path to the data file, relative to this script\'s location in src/
DATA_FILE_PATH = '../features/combined_cleaned_data.pkl'

# --- Helper Functions ---
def load_data(file_path):
    """Loads data from a pickle file."""
    # Construct absolute path from script's location
    absolute_file_path = os.path.join(os.path.dirname(__file__), file_path)
    if not os.path.exists(absolute_file_path):
        print(f"Error: Data file not found at {absolute_file_path}")
        return None
    try:
        with open(absolute_file_path, 'rb') as f:
            data = pickle.load(f)
        if not isinstance(data, pd.DataFrame):
            print(f"Error: Loaded data from {absolute_file_path} is not a Pandas DataFrame.")
            return None
        print(f"Data loaded successfully from {absolute_file_path}")
        return data
    except Exception as e:
        print(f"Error loading data from {absolute_file_path}: {e}")
        return None

def get_user_choice(prompt, choices):
    """Gets a valid choice from the user from a list."""
    if not choices:
        print("No choices available.")
        return None
    print(f"\\n{prompt}")
    for i, choice in enumerate(choices):
        print(f"{i+1}. {choice}")
    while True:
        try:
            user_input = input(f"Enter your choice (number 1-{len(choices)}): ")
            choice_idx = int(user_input) - 1
            if 0 <= choice_idx < len(choices):
                return choices[choice_idx]
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(choices)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except EOFError:
            print("\\nInput cancelled.")
            return None


# --- Main Script ---
def main():
    df = load_data(DATA_FILE_PATH)
    if df is None:
        return

    # --- Subject Selection ---
    if SUBJECT_COLUMN not in df.columns:
        print(f"Error: Subject column \'{SUBJECT_COLUMN}\' not found in DataFrame.")
        print(f"Available columns: {df.columns.tolist()}")
        print("Please update SUBJECT_COLUMN at the top of the script if needed.")
        return

    subjects = sorted(df[SUBJECT_COLUMN].unique())
    if not subjects:
        print(f"No unique subjects found in column \'{SUBJECT_COLUMN}\'.")
        return

    selected_subject = get_user_choice(f"Select a subject ID from \'{SUBJECT_COLUMN}\':", subjects)
    if selected_subject is None: return # User cancelled

    subject_data = df[df[SUBJECT_COLUMN] == selected_subject].copy()

    if subject_data.empty:
        print(f"No data found for subject {selected_subject}.")
        return

    # --- Timestamp Column Selection ---
    all_columns = subject_data.columns.tolist()
    timestamp_column_choices = all_columns + ["Use DataFrame Index"]
    chosen_timestamp_option = get_user_choice("Select the column to use for timestamps (or use index):", timestamp_column_choices)

    if chosen_timestamp_option is None: return # User cancelled
    
    actual_timestamp_column = None
    if chosen_timestamp_option != "Use DataFrame Index":
        actual_timestamp_column = chosen_timestamp_option
    else:
        print("Using DataFrame index for the x-axis.")


    # --- Sensor Selection ---
    excluded_cols_for_sensors = [SUBJECT_COLUMN]
    if actual_timestamp_column and actual_timestamp_column in subject_data.columns: # Check if it's a valid column name
        excluded_cols_for_sensors.append(actual_timestamp_column)
    
    potential_sensor_columns = sorted([
        col for col in subject_data.columns
        if pd.api.types.is_numeric_dtype(subject_data[col]) and \
           col not in excluded_cols_for_sensors
    ])

    if not potential_sensor_columns:
        print(f"No numeric sensor columns found for subject {selected_subject} (excluding: {', '.join(excluded_cols_for_sensors)}).")
        print(f"Columns in subject's data: {subject_data.columns.tolist()}")
        return

    selected_sensor = get_user_choice(f"Select a sensor to plot for subject {selected_subject}:", potential_sensor_columns)
    if selected_sensor is None: return # User cancelled

    # --- Data for Plotting ---
    x_data = None
    x_label = "Index" # Default x-axis label

    if actual_timestamp_column and actual_timestamp_column in subject_data.columns:
        try:
            # Attempt to convert to datetime, handling potential errors
            subject_data[actual_timestamp_column] = pd.to_datetime(subject_data[actual_timestamp_column], errors='coerce')
            if subject_data[actual_timestamp_column].isnull().all(): # All values failed to parse
                 print(f"Warning: Column \'{actual_timestamp_column}\' could not be converted to datetime (all values are NaT). Using index for x-axis.")
            else:
                x_data = subject_data[actual_timestamp_column]
                x_label = actual_timestamp_column
        except Exception as e:
            print(f"Warning: Could not convert \'{actual_timestamp_column}\' to datetime: {e}. Using index for x-axis.")
    
    if x_data is None: # Fallback to index if timestamp is not used or failed
        x_data = subject_data.index
        if actual_timestamp_column: # Only print warning if a timestamp column was chosen but failed/not found
            if actual_timestamp_column not in subject_data.columns: # Should not happen due to earlier checks, but good for safety
                print(f"Warning: Chosen timestamp column \'{actual_timestamp_column}\' not found. Using DataFrame index for x-axis.")
            # If it was found but failed conversion, previous messages cover it.
        # No message needed if "Use DataFrame Index" was explicitly chosen and actual_timestamp_column is None


    y_data = subject_data[selected_sensor]

    # --- Plotting ---
    plt.figure(figsize=(15, 7))
    plt.plot(x_data, y_data, label=selected_sensor)
    
    plt.title(f"Data for Subject: {selected_subject}, Sensor: {selected_sensor}")
    plt.xlabel(x_label)
    plt.ylabel("Sensor Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout() # Adjust plot to prevent labels from being cut off
    
    # --- Saving the Plot ---
    # Determine the root directory of the workspace (one level up from src/)
    script_dir = os.path.dirname(__file__)
    workspace_root = os.path.abspath(os.path.join(script_dir, os.pardir))
    output_filename = "combined_cleaned_test_plot.png"
    full_output_path = os.path.join(workspace_root, output_filename)

    try:
        plt.savefig(full_output_path)
        print(f"Plot saved successfully to {full_output_path}")
    except Exception as e:
        print(f"Error saving plot to {full_output_path}: {e}")
    
    # For very high-frequency data, you might want to improve plotting performance or clarity.
    # Consider plotting every Nth point:
    # plt.plot(x_data[::10], y_data[::10], label=selected_sensor) # Plot every 10th point
    #
    # Or, if you have a proper datetime x-axis, resample the data:
    # if pd.api.types.is_datetime64_any_dtype(x_data):
    #     plot_df = pd.DataFrame({x_label: x_data, selected_sensor: y_data}).set_index(x_label)
    #     # Example: Resample to 1-second mean. Adjust '1S' as needed (e.g., '100ms', '5T' for 5 minutes)
    #     resampled_data = plot_df[selected_sensor].resample('1S').mean() 
    #     plt.plot(resampled_data.index, resampled_data.values, label=f"{selected_sensor} (resampled to 1s mean)")
    # else:
    #     plt.plot(x_data, y_data, label=selected_sensor)


    print("Displaying plot. Close the plot window to continue/exit the script.")
    plt.show()

if __name__ == "__main__":
    main()
