# src/utils.py

import logging
import os
import random
import numpy as np
import torch
import pickle
import re
from torch.utils.data import Dataset
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.dates as mdates
import random 

# --- Logging Setup ---
# Note: Logging is typically configured once in the main orchestrator script
# This function can be called by main_pipeline.py
def setup_logging(log_file='pipeline.log', level=logging.INFO):
    """Configures logging to file and console."""
    # Ensure the logger is clear of previous handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ]
    )
    logging.info("--- Logging Initialized ---")

# --- Seeding ---
def set_seed(seed):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Optional: These can impact performance
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    logging.info(f"Random seeds set to {seed}")

# --- File Handling ---
def save_pickle(data, file_path):
    """Saves data to a pickle file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        logging.info(f"Saved data to {file_path}")
    except Exception as e:
        logging.error(f"Error saving pickle file {file_path}: {e}", exc_info=True)
        raise

def load_pickle(file_path):
    """Loads data from a pickle file."""
    if not os.path.exists(file_path):
         logging.error(f"Pickle file not found: {file_path}")
         raise FileNotFoundError(f"Pickle file not found: {file_path}")
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        logging.info(f"Loaded data from {file_path}")
        return data
    except Exception as e:
        logging.error(f"Error loading pickle file {file_path}: {e}", exc_info=True)
        raise

def save_pytorch_artifact(data, file_path):
    """Saves PyTorch models or tensors."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        torch.save(data, file_path)
        logging.info(f"Saved PyTorch artifact to {file_path}")
    except Exception as e:
        logging.error(f"Error saving PyTorch artifact {file_path}: {e}", exc_info=True)
        raise

def load_pytorch_artifact(file_path, map_location='cpu'):
    """Loads PyTorch models or tensors."""
    if not os.path.exists(file_path):
         logging.error(f"PyTorch artifact file not found: {file_path}")
         raise FileNotFoundError(f"PyTorch artifact file not found: {file_path}")
    try:
        # map_location helps load models saved on GPU onto CPU if needed
        data = torch.load(file_path, map_location=map_location)
        logging.info(f"Loaded PyTorch artifact from {file_path}")
        return data
    except Exception as e:
        logging.error(f"Error loading PyTorch artifact {file_path}: {e}", exc_info=True)
        raise

def save_numpy(data, file_path):
    """Saves a NumPy array to a .npy file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        np.save(file_path, data)
        logging.info(f"Saved NumPy array to {file_path} (shape: {data.shape}, dtype: {data.dtype})")
    except Exception as e:
        logging.error(f"Error saving NumPy array {file_path}: {e}", exc_info=True)
        raise

def load_numpy(file_path, allow_pickle=True): # Add allow_pickle argument with default False
    """Loads a NumPy array from a .npy file."""
    if not os.path.exists(file_path):
         logging.error(f"NumPy file not found: {file_path}")
         raise FileNotFoundError(f"NumPy file not found: {file_path}")
    try:
        # Pass allow_pickle to np.load
        data = np.load(file_path, allow_pickle=allow_pickle)
        logging.info(f"Loaded NumPy array from {file_path} (shape: {data.shape}, dtype: {data.dtype})")
        return data
    except Exception as e:
        logging.error(f"Error loading NumPy array {file_path}: {e}", exc_info=True)
        raise

# --- Other Utilities ---
def extract_subject_id(filename, pattern=r"^(OutSense-\d+)_"):
    """
    Extracts a subject identifier (e.g., 'OutSense-498') from a filename
    using a regular expression.
    """
    basename = os.path.basename(filename)
    match = re.match(pattern, basename)
    if match:
        subject_id = match.group(1)
        return subject_id
    else:
        # Fallback: Use the part of the filename before the first underscore
        parts = basename.split('_')
        if parts:
            fallback_id = parts[0]
            logging.debug(f"Could not extract subject ID using pattern from '{basename}'. Using fallback: '{fallback_id}'")
            return fallback_id
        else:
            logging.error(f"Could not extract subject ID or fallback from filename: '{basename}'")
            return None # Indicate failure
        
def get_random_color_hex():
    """Generates a random hex color string."""
    # Ensure a good range of colors, avoiding too dark/light if possible, though this is basic random
    r = random.randint(50, 200)
    g = random.randint(50, 200)
    b = random.randint(50, 200)
    return f"#{r:02x}{g:02x}{b:02x}"
        
def plot_timeseries_to_pdf(
    dataframe: pd.DataFrame,
    pdf_pages_object: PdfPages,
    title_base: str,
    subject_labels_df: pd.DataFrame = None,
    columns_to_plot: list = None,
    chunk_duration_minutes: int = 5,
    plot_label_colors: dict = None
):
    """
    Plots a timeseries DataFrame to a multi-page PDF, with each page
    representing a fixed time chunk and overlaying activity labels.

    Args:
        dataframe (pd.DataFrame): DataFrame with a DatetimeIndex and numeric columns to plot.
        pdf_pages_object (PdfPages): Matplotlib PdfPages object to save the figures to.
        title_base (str): Base string for the plot titles (e.g., "Subject X - Stage Y - Sensor Z").
        subject_labels_df (pd.DataFrame, optional): DataFrame with 'Real_Start_Time', 
                                                    'Real_End_Time', and 'Label' columns.
                                                    Timestamps should be timezone-naive and match dataframe.index.
        columns_to_plot (list, optional): List of column names to plot. 
                                          If None, all numeric columns are plotted.
        chunk_duration_minutes (int, optional): Duration in minutes for each plot page. Defaults to 5.
        plot_label_colors (dict, optional): Dictionary mapping activity labels to colors for axvspan.
                                            Example: {'Walking': 'lightblue', 'Sitting': 'lightgreen'}
    """
def plot_timeseries_to_pdf(
    dataframe: pd.DataFrame,
    pdf_pages_object: PdfPages,
    title_base: str,
    subject_labels_df: pd.DataFrame = None,
    columns_to_plot: list = None,
    chunk_duration_minutes: int = 5,
    plot_label_colors: dict = None # Predefined colors from config
):
    """
    Plots a timeseries DataFrame to a multi-page PDF, with each page
    representing a fixed time chunk and overlaying activity labels.
    Random colors are assigned to labels not in plot_label_colors for this PDF.
    """
    # This dictionary will store colors for labels not in plot_label_colors,
    # ensuring consistency within this single PDF generation.
    local_activity_colors = {}

    if not isinstance(dataframe.index, pd.DatetimeIndex):
        logging.warning(f"Plotting function for '{title_base}': DataFrame does not have a DatetimeIndex. Skipping plot.")
        return

    if dataframe.empty:
        logging.warning(f"Plotting function for '{title_base}': DataFrame is empty. Skipping plot.")
        return

    if chunk_duration_minutes <= 0:
        logging.warning(f"Plotting function for '{title_base}': chunk_duration_minutes must be positive. Skipping plot.")
        return

    if columns_to_plot is None or not columns_to_plot:
        cols_to_plot_internal = dataframe.select_dtypes(include=np.number).columns.tolist()
    else:
        cols_to_plot_internal = [col for col in columns_to_plot if col in dataframe.columns and pd.api.types.is_numeric_dtype(dataframe[col])]

    if not cols_to_plot_internal:
        logging.warning(f"Plotting function for '{title_base}': No valid numeric columns to plot found. Skipping.")
        return

    df_to_plot = dataframe.copy()
    if df_to_plot.index.tz is not None:
        df_to_plot.index = df_to_plot.index.tz_localize(None)

    labels_df_processed = None
    if subject_labels_df is not None and not subject_labels_df.empty:
        labels_df_processed = subject_labels_df.copy()
        for col_name in ['Real_Start_Time', 'Real_End_Time']:
            if col_name in labels_df_processed.columns and \
               pd.api.types.is_datetime64_any_dtype(labels_df_processed[col_name]):
                if labels_df_processed[col_name].dt.tz is not None:
                    labels_df_processed[col_name] = labels_df_processed[col_name].dt.tz_localize(None)
            else:
                try:
                    labels_df_processed[col_name] = pd.to_datetime(labels_df_processed[col_name])
                    if labels_df_processed[col_name].dt.tz is not None: # Check again
                         labels_df_processed[col_name] = labels_df_processed[col_name].dt.tz_localize(None)
                except Exception as e:
                    logging.warning(f"Plotting function for '{title_base}': Could not process time column '{col_name}' in labels_df. Error: {e}")
                    labels_df_processed = None
                    break
    
    chunk_delta = pd.Timedelta(minutes=chunk_duration_minutes)
    current_plot_time = df_to_plot.index.min()
    overall_data_end_time = df_to_plot.index.max()
    page_count = 0
    max_pages = 750

    while current_plot_time <= overall_data_end_time:
        page_count += 1
        if page_count > max_pages:
            logging.warning(f"Plotting function for '{title_base}': Reached {max_pages} page limit. Stopping PDF generation for this item.")
            break

        chunk_start_time = current_plot_time
        chunk_end_time = chunk_start_time + chunk_delta
        effective_chunk_end = min(chunk_end_time, overall_data_end_time + pd.Timedelta(nanoseconds=1))
        chunk_df = df_to_plot[(df_to_plot.index >= chunk_start_time) & (df_to_plot.index < effective_chunk_end)]

        if chunk_df.empty:
            if current_plot_time >= overall_data_end_time: break
            current_plot_time += chunk_delta
            continue

        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        active_labels_in_chunk_details = [] # Store (name, color) tuples for legend

        for col in cols_to_plot_internal:
            if col in chunk_df.columns:
                ax.plot(chunk_df.index, chunk_df[col], label=col, linewidth=0.5, alpha=0.9)

        if labels_df_processed is not None and not labels_df_processed.empty and \
           all(c in labels_df_processed.columns for c in ['Real_Start_Time', 'Real_End_Time', 'Label']):
            overlapping_labels = labels_df_processed[
                (labels_df_processed['Real_Start_Time'] < effective_chunk_end) &
                (labels_df_processed['Real_End_Time'] > chunk_start_time)
            ]
            for _, label_row in overlapping_labels.iterrows():
                label_plot_start = max(pd.to_datetime(label_row['Real_Start_Time']), chunk_start_time)
                label_plot_end = min(pd.to_datetime(label_row['Real_End_Time']), chunk_df.index.max() if not chunk_df.empty else effective_chunk_end)
                if label_plot_start >= label_plot_end: continue

                activity_name = str(label_row['Label'])
                color_for_activity = None
                if plot_label_colors and activity_name in plot_label_colors:
                    color_for_activity = plot_label_colors[activity_name]
                elif activity_name in local_activity_colors:
                    color_for_activity = local_activity_colors[activity_name]
                else:
                    color_for_activity = get_random_color_hex()
                    local_activity_colors[activity_name] = color_for_activity
                
                if color_for_activity is None: color_for_activity = 'lightgrey' # Fallback

                ax.axvspan(label_plot_start, label_plot_end, color=color_for_activity, alpha=0.25, zorder=0)
                # Add to details for legend if not already added for this chunk
                if not any(d[0] == activity_name for d in active_labels_in_chunk_details):
                    active_labels_in_chunk_details.append((activity_name, color_for_activity))
        
        sensor_handles, sensor_labels_text = ax.get_legend_handles_labels()
        activity_proxies = []
        activity_legend_labels_text = []

        # Sort active_labels_in_chunk_details by name for consistent legend order
        active_labels_in_chunk_details.sort(key=lambda x: x[0]) 
        for activity_name, color_val in active_labels_in_chunk_details:
            activity_proxies.append(plt.Rectangle((0, 0), 1, 1, fc=color_val, alpha=0.3))
            activity_legend_labels_text.append(f"Act: {activity_name}")
        
        if sensor_handles or activity_proxies:
            all_handles = sensor_handles + activity_proxies
            all_labels_text = sensor_labels_text + activity_legend_labels_text
            num_legend_items = len(all_handles)
            legend_cols = 1
            if num_legend_items > 5: legend_cols = 2
            if num_legend_items > 12: legend_cols = 3
            if num_legend_items > 20: legend_cols = 4
            ax.legend(all_handles, all_labels_text, loc='upper right', fontsize='xx-small', ncol=legend_cols, framealpha=0.7)

        start_str = chunk_start_time.strftime('%Y-%m-%d %H:%M:%S')
        end_str_actual = (chunk_df.index.max() if not chunk_df.empty else chunk_start_time).strftime('%H:%M:%S')
        plot_title_detail = f"{title_base} | {start_str} to {end_str_actual} (Page {page_count})"
        
        ax.set_title(plot_title_detail, fontsize=9)
        ax.set_xlabel("Time", fontsize=8)
        ax.set_ylabel("Sensor Value", fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10))
        fig.autofmt_xdate(rotation=30, ha='right')
        ax.grid(True, linestyle=':', alpha=0.5)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        pdf_pages_object.savefig(fig, orientation='landscape')
        plt.close(fig)

        if chunk_start_time == overall_data_end_time and (chunk_df.empty or chunk_df.index.max() == overall_data_end_time):
            break
        current_plot_time += chunk_delta
        if current_plot_time > overall_data_end_time and (chunk_df.empty or chunk_df.index.max() == overall_data_end_time):
            break

# --- CuPy Check ---
# Moved CUPY_AVAILABLE check to the top level where needed,
# but keeping a function here might be useful if multiple modules need it.
_CUPY_AVAILABLE = None
def check_cupy_availability():
    """Checks for CuPy availability once and caches the result."""
    global _CUPY_AVAILABLE
    if _CUPY_AVAILABLE is None:
        try:
            import cupy as cp
            if cp.is_available():
                _CUPY_AVAILABLE = True
                logging.info("CuPy available.")
            else:
                _CUPY_AVAILABLE = False
                logging.warning("CuPy installed but no compatible GPU found.")
        except ImportError:
            _CUPY_AVAILABLE = False
            logging.info("CuPy not found.")
    return _CUPY_AVAILABLE

class SensorDataset(Dataset):
    """
    PyTorch Dataset class for handling sensor data windows.
    It stores features (X) and optionally labels (y) as tensors.
    """
    def __init__(self, X, y=None):
        """
        Initializes the dataset.

        Args:
            X (np.ndarray or torch.Tensor): Data windows (features).
                                             Expected shape (N, C, L) for Conv1D.
            y (np.ndarray or torch.Tensor, optional): Labels corresponding to windows.
                                                      Expected shape (N,). Defaults to None.
        """
        # Convert features to FloatTensor if they aren't already
        if isinstance(X, np.ndarray):
            self.X = torch.tensor(X, dtype=torch.float32)
        elif torch.is_tensor(X):
            self.X = X.float()
        else:
            raise TypeError(f"Input X must be a NumPy array or PyTorch Tensor, got {type(X)}")

        # Convert labels to LongTensor if provided
        if y is not None:
            if isinstance(y, np.ndarray):
                self.y = torch.tensor(y, dtype=torch.long)
            elif torch.is_tensor(y):
                 # Ensure correct dtype if tensor is passed
                self.y = y.long()
            else:
                raise TypeError(f"Input y must be a NumPy array or PyTorch Tensor, got {type(y)}")

            if self.X.shape[0] != self.y.shape[0]:
                raise ValueError(f"Mismatch in number of samples between X ({self.X.shape[0]}) and y ({self.y.shape[0]})")
        else:
            self.y = None

    def __len__(self):
        """Returns the total number of windows (samples) in the dataset."""
        return len(self.X)

    def __getitem__(self, idx):
        """
        Retrieves a single window and its corresponding label (if available).
        """
        if self.y is not None:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx]