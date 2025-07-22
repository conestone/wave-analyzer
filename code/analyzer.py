#!/usr/bin/env python3
"""
Wave Event Analyzer

A comprehensive tool for analyzing wave events from boat wave time series data.
This module identifies significant wave peaks and bottoms, groups them into events,
and generates detailed visualizations and summary reports organized by date.

Author: Oliver Konold
Date: 2025
Version: 1.1 - Integrated DataLoader
License: MIT
"""

import os
import sys
import subprocess
import importlib
import yaml
import pandas as pd
import numpy as np
from datetime import timedelta
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from tqdm import tqdm
import shutil


def check_and_install_packages():
    """
    Check if required packages are installed and install them if missing.
    
    This function verifies that all necessary Python packages are available
    and automatically installs any missing dependencies using pip.
    
    Raises:
        SystemExit: If package installation fails
    """
    required_packages = {
        'yaml': 'PyYAML',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'matplotlib': 'matplotlib',
        'scipy': 'scipy',
        'tqdm': 'tqdm'
    }
    
    missing_packages = []
    
    print("Checking required packages...")
    for module_name, package_name in required_packages.items():
        try:
            importlib.import_module(module_name)
            print(f"[OK] {package_name} is installed")
        except ImportError:
            print(f"[MISSING] {package_name} is missing")
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"\nInstalling missing packages: {', '.join(missing_packages)}")
        for package in missing_packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"[OK] Successfully installed {package}")
            except subprocess.CalledProcessError:
                print(f"[ERROR] Failed to install {package}")
                sys.exit(1)
        print("All packages installed successfully!\n")
    else:
        print("All required packages are installed!\n")


class DataLoader:
    """
    Data loader for processing multiple CSV files containing wave measurement data.
    
    This class handles the loading and processing of raw CSV files containing
    wave measurements with multiple samples per timestamp, expanding them into
    a proper time series format.
    
    Attributes:
        base_path (Path): Base directory path for data operations
        freq_hz (int): Sampling frequency in Hz
        dt (pd.Timedelta): Time delta between samples
        sample_cols (list): Column names for sample data
        df_timeseries (pd.DataFrame): Processed time series data
    """
    
    def __init__(self, base_path: str, freq_hz: int = 8):
        """
        Initialize the DataLoader.
        
        Args:
            base_path (str): Base directory path containing input folder
            freq_hz (int, optional): Sampling frequency in Hz. Defaults to 8.
        """
        self.base_path = Path(base_path)
        self.freq_hz = freq_hz
        self.dt = pd.to_timedelta(1 / freq_hz, unit="s")
        self.sample_cols = [f"sample{i+1}" for i in range(32)]
        self.df_timeseries = None

    def load_all(self):
        """
        Load and process all CSV files from the input directory.
        
        This method finds all CSV files in the input directory, loads them,
        processes the sample data into proper time series format, and combines
        all files into a single pivoted DataFrame.
        
        Returns:
            pd.DataFrame: Processed time series data with timestamps as index
                         and directions as columns
                         
        Raises:
            FileNotFoundError: If no CSV files are found in the input directory
        """
        input_dir = self.base_path / "input"
        all_csv_files = list(input_dir.glob("*.csv"))

        if not all_csv_files:
            raise FileNotFoundError(f"No CSV files found in {input_dir}")

        all_series = []

        for file in all_csv_files:
            df_raw = pd.read_csv(file, header=None)
            df_raw.columns = ["direction", "s", "b", "date"] + self.sample_cols
            df_raw[self.sample_cols] = df_raw[self.sample_cols].astype(np.float32)

            all_series.append(self.expand_samples(df_raw))

        df_concat = pd.concat(all_series)
        df_pivot = df_concat.pivot(index="timestamp", columns="direction", values="value").sort_index()

        self.df_timeseries = df_pivot
        return df_pivot

    def expand_samples(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Expand sample columns into individual time series entries.
        
        This method takes raw data with multiple samples per row and expands
        it into individual timestamp-value pairs, properly spacing the timestamps
        according to the sampling frequency.
        
        Args:
            df_raw (pd.DataFrame): Raw data with sample columns
            
        Returns:
            pd.DataFrame: Expanded time series data with individual timestamps
        """
        directions = df_raw["direction"].unique()
        all_expanded = []

        for direction in directions:
            subset = df_raw[df_raw["direction"] == direction]
            dates = np.repeat(subset["date"].values, len(self.sample_cols))
            sample_idx = np.tile(np.arange(32), len(subset))
            timestamps = pd.to_datetime(dates) + sample_idx * self.dt

            values = subset[self.sample_cols].values.ravel()

            series = pd.DataFrame({
                "timestamp": timestamps,
                "value": values,
                "direction": direction
            })

            all_expanded.append(series)

        return pd.concat(all_expanded)


class WaveEventAnalyzer:
    """
    A comprehensive analyzer for wave events in time series data.
    
    This class processes wave amplitude data to identify significant wave events,
    group them temporally, and generate detailed analysis outputs including
    plots and summary statistics organized by date.
    
    Attributes:
        config (dict): Configuration parameters loaded from YAML file
        base_path (str): Base directory for input/output operations
        data_path (str): Path to input CSV file (legacy, now uses DataLoader)
        output_base_dir (str): Base directory for plot outputs
        tables_base_dir (str): Base directory for table outputs
        column (str): Name of the wave amplitude column to analyze
        fs (float): Sampling rate in Hz
        threshold (float): Amplitude threshold for significant events
        window_samples (int): Window size in samples for peak-bottom pairing
        event_gap (pd.Timedelta): Minimum time gap between separate events
        padding (timedelta): Time padding around events for plotting
        data_loader (DataLoader): Instance of DataLoader for handling multiple CSV files
        df (pd.DataFrame): Main dataset with processed time series
        wave (np.ndarray): Wave amplitude values as numpy array
        peaks (np.ndarray): Indices of detected peaks
        bottoms (np.ndarray): Indices of detected bottoms
        significant_df (pd.DataFrame): DataFrame containing significant peak-bottom pairs
        date_folders (dict): Mapping of date strings to output folder paths
    """
    
    def __init__(self, config_path):
        """
        Initialize the WaveEventAnalyzer with configuration from YAML file.
        
        Args:
            config_path (str): Path to the YAML configuration file
            
        Raises:
            FileNotFoundError: If configuration file doesn't exist
            KeyError: If required configuration keys are missing
        """
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Initialize paths from configuration
        # self.base_path = self.config["base_path"]
        self.base_path = os.path.dirname(os.path.abspath(config_path))

        
        # Initialize DataLoader for handling multiple CSV files
        self.data_loader = DataLoader(
            base_path=self.base_path,
            freq_hz=self.config.get("sampling_rate_hz", 8) # fallback value if Hz is not defined
        )
        
        # Legacy support for single CSV file (kept for backward compatibility)
        if "input_file" in self.config:
            self.data_path = os.path.join(self.base_path, "input", self.config["input_file"])
        else:
            self.data_path = None
        
        # Base output directories - date subfolders will be created later
        self.output_base_dir = os.path.join(self.base_path, "output/plots")
        self.tables_base_dir = os.path.join(self.base_path, "output/tables")
        
        # These will be set after loading data when we know the date range
        self.output_dir = None
        self.tables_dir = None

        # Load analysis parameters from configuration
        self.vertical = self.config["vertical_variable"]
        self.lateral = self.config["lateral_variable"]
        self.fs = self.config["sampling_rate_hz"]
        self.threshold = self.config["amplitude_threshold"]
        self.window_samples = int(self.config["significant_window_sec"] * self.fs)
        self.event_gap = pd.Timedelta(seconds=self.config["event_gap_sec"])
        self.padding = timedelta(seconds=self.config["plot_padding_sec"])

        # Initialize data containers
        self.df = None
        self.wave = None
        self.peaks = None
        self.bottoms = None
        self.significant_df = None
        self.date_folders = {}  # To store date-specific output folders

    def create_date_folders(self):
        """
        Create date-specific subfolders for each unique date in the dataset.
        
        This method extracts all unique dates from the time series data and creates
        corresponding subfolders in both plots and tables output directories.
        Any existing folders with the same dates are deleted first to ensure
        clean analysis results.
        
        Side effects:
            - Deletes existing date folders if they exist
            - Creates new empty date folders
            - Populates self.date_folders dictionary
        """
        print("Creating date-specific folders...")
        unique_dates = self.df.index.date
        unique_dates = pd.Series(unique_dates).drop_duplicates().sort_values()
        
        for date in unique_dates:
            date_str = date.strftime("%Y%m%d")
            
            # Define paths for this date
            date_plots_dir = os.path.join(self.output_base_dir, date_str)
            date_tables_dir = os.path.join(self.tables_base_dir, date_str)
            
            # Check if folders exist and delete them
            if os.path.exists(date_plots_dir):
                print(f"[INFO] Deleting existing plots folder: {date_plots_dir}")
                shutil.rmtree(date_plots_dir)
            
            if os.path.exists(date_tables_dir):
                print(f"[INFO] Deleting existing tables folder: {date_tables_dir}")
                shutil.rmtree(date_tables_dir)
            
            # Create fresh folders
            os.makedirs(date_plots_dir, exist_ok=True)
            os.makedirs(date_tables_dir, exist_ok=True)
            
            self.date_folders[date_str] = {
                'plots': date_plots_dir,
                'tables': date_tables_dir
            }
        
        print(f"[OK] Created fresh folders for {len(unique_dates)} unique dates: {list(self.date_folders.keys())}")

    def load_data(self):
        """
        Load and preprocess the wave data using DataLoader or legacy CSV method.
        
        This method now supports two loading modes:
        1. New mode: Uses DataLoader to process multiple CSV files from input directory
        2. Legacy mode: Loads single CSV file (for backward compatibility)
        
        The method automatically detects which mode to use based on configuration
        and data availability.
        
        Raises:
            ValueError: If time column is not found in the data
            FileNotFoundError: If no input files are found
            
        Side effects:
            - Sets self.df with processed DataFrame
            - Sets self.wave with amplitude values
            - Creates date-specific output folders
        """
        print("Loading data...")
        
        try:
            # Try using DataLoader for multiple CSV files first
            df_pivot = self.data_loader.load_all()

            # Check that both vertical and lateral variables are present
            required_columns = [self.vertical, self.lateral]
            missing_cols = [col for col in required_columns if col not in df_pivot.columns]
            if missing_cols:
                available_cols = list(df_pivot.columns)
                raise ValueError(f"Missing required columns: {missing_cols}. Available columns: {available_cols}")

            # Create DataFrame with both vertical and lateral variables
            df = pd.DataFrame({col: df_pivot[col] for col in required_columns})
            df.index = df_pivot.index
            print(f"[OK] Loaded data using DataLoader with columns: {', '.join(required_columns)}")

                
        except FileNotFoundError:
            # Fall back to legacy single CSV file loading
            if self.data_path and os.path.exists(self.data_path):
                print("DataLoader failed, falling back to legacy CSV loading...")
                df = pd.read_csv(self.data_path)
                df.columns = [col.strip().lower() for col in df.columns]

                # Auto-detect time column (supports German 'zeitstempel' and English 'timestamp')
                if "zeitstempel" in df.columns:
                    self.time_col = "zeitstempel"
                elif "timestamp" in df.columns:
                    self.time_col = "timestamp"
                else:
                    raise ValueError(f"Time column not found. Available: {df.columns.tolist()}")

                # Convert to datetime and sort chronologically
                df[self.time_col] = pd.to_datetime(df[self.time_col], format="%Y-%m-%d %H:%M:%S.%f")
                df = df.sort_values(self.time_col).reset_index(drop=True)
                df.set_index(self.time_col, inplace=True)
                print(f"[OK] Loaded data using legacy CSV method")
                
            else:
                raise FileNotFoundError("No CSV files found in input directory and no legacy CSV file specified")

        # Initialize peak and bottom identification columns
        df["is_peak"] = False
        df["is_bottom"] = False

        self.df = df
        self.wave = df[self.vertical].values
        print(f"[OK] Loaded {len(df)} data points")
        
        # Create date-specific folders after loading data
        self.create_date_folders()

    def detect_peaks_and_bottoms(self):
        """
        Detect all peaks and bottoms in the wave data using scipy.signal.find_peaks.
        
        This method identifies local maxima (peaks) and local minima (bottoms)
        in the wave amplitude data and marks them in the DataFrame.
        
        Side effects:
            - Sets self.peaks with peak indices
            - Sets self.bottoms with bottom indices  
            - Updates DataFrame with is_peak and is_bottom flags
        """
        print("Detecting peaks and bottoms...")
        self.peaks, _ = find_peaks(self.wave)
        self.bottoms, _ = find_peaks(-self.wave)  # Invert signal to find minima
        
        # Mark peaks and bottoms in the DataFrame
        self.df.loc[self.df.index[self.peaks], "is_peak"] = True
        self.df.loc[self.df.index[self.bottoms], "is_bottom"] = True
        
        print(f"[OK] Found {len(self.peaks)} peaks and {len(self.bottoms)} bottoms")

    def find_significant_pairs(self):
        """
        Find significant peak-bottom pairs based on amplitude threshold.
        
        This method examines each detected peak and searches for nearby bottoms
        within a specified time window. Pairs with amplitude differences exceeding
        the threshold are considered significant wave events.
        
        The algorithm:
        1. For each peak, define a search window around it
        2. Find all bottoms within that window
        3. Calculate amplitude differences between peak and each bottom
        4. Keep pairs where the difference exceeds the configured threshold
        
        Side effects:
            - Sets self.significant_df with significant peak-bottom pairs
            - Shows progress bar during processing
        """
        print("Finding significant peak-bottom pairs...")
        pairs = []
        
        # Progress bar for peak analysis
        with tqdm(total=len(self.peaks), desc="Analyzing peaks", unit="peak") as pbar:
            for p_idx in self.peaks:
                peak_val = self.wave[p_idx]
                peak_time = self.df.index[p_idx]

                # Define search window around the peak
                min_idx = max(p_idx - self.window_samples, 0)
                max_idx = min(p_idx + self.window_samples, len(self.wave) - 1)
                candidate_bottoms = self.bottoms[(self.bottoms >= min_idx) & (self.bottoms <= max_idx)]

                # Skip if no candidate bottoms found in window
                if candidate_bottoms.size == 0:
                    pbar.update(1)
                    continue

                # Calculate amplitude differences and filter by threshold
                bottom_vals = self.wave[candidate_bottoms]
                diffs = np.abs(peak_val - bottom_vals)
                valid = diffs > self.threshold

                # Store significant pairs
                for b_idx, diff in zip(candidate_bottoms[valid], diffs[valid]):
                    pairs.append({
                        "peak_time": peak_time,
                        "bottom_time": self.df.index[b_idx],
                        "peak_val": peak_val,
                        "bottom_val": self.wave[b_idx],
                        "amplitude_diff": diff
                    })
                
                pbar.update(1)

        # Convert to DataFrame and sort by time
        self.significant_df = pd.DataFrame(pairs)
        self.significant_df = self.significant_df.sort_values("peak_time").reset_index(drop=True)
        print(f"[OK] Found {len(self.significant_df)} significant peak-bottom pairs")

    def group_events(self):
        """
        Group significant pairs into events based on time gaps, with event numbering restarting for each date.
        
        This method analyzes the temporal distribution of significant peak-bottom
        pairs and groups nearby pairs into discrete wave events. Events are
        separated when the time gap between consecutive peaks exceeds the
        configured event_gap threshold.
        
        Event numbering starts from 1 for each date, so each day has events numbered 1, 2, 3, etc.
        
        Side effects:
            - Adds event_id column to self.significant_df
            - Event IDs start from 1 for each date and increment sequentially
            
        Returns:
            None: Updates significant_df in place
        """
        print("Grouping events by date...")
        if len(self.significant_df) == 0:
            print("[WARNING] No significant pairs found to group")
            return
        
        # Add date column to help with grouping
        self.significant_df['date'] = self.significant_df['peak_time'].dt.date
        
        # Initialize list to store all event IDs
        all_event_ids = []
        
        # Group by date and process each date separately
        for date, date_group in self.significant_df.groupby('date'):
            date_indices = date_group.index.tolist()
            
            # Sort by peak_time to ensure chronological order within the date
            date_group_sorted = date_group.sort_values('peak_time')
            
            # Initialize event grouping for this date - start with event 1
            date_event_ids = [1]  # Start with event 1 for this date
            event_counter = 1
            
            # Group consecutive peaks into events based on time gaps within this date
            for i in range(1, len(date_group_sorted)):
                current_idx = date_group_sorted.index[i]
                prev_idx = date_group_sorted.index[i-1]
                
                current_time = self.significant_df.loc[current_idx, "peak_time"]
                prev_time = self.significant_df.loc[prev_idx, "peak_time"]
                
                delta = current_time - prev_time
                if delta > self.event_gap:
                    event_counter += 1
                date_event_ids.append(event_counter)
            
            # Create a mapping from original indices to event IDs for this date
            for i, original_idx in enumerate(date_group_sorted.index):
                # Store the event ID with date prefix to make it unique across dates
                date_str = date.strftime("%Y%m%d")
                unique_event_id = f"{date_str}_{date_event_ids[i]:02d}"
                all_event_ids.append((original_idx, unique_event_id))
        
        # Apply event IDs back to the original DataFrame
        for original_idx, event_id in all_event_ids:
            self.significant_df.loc[original_idx, "event_id"] = event_id
        
        # Clean up the temporary date column
        self.significant_df.drop('date', axis=1, inplace=True)
        
        num_events = len(self.significant_df["event_id"].unique())
        print(f"[OK] Grouped into {num_events} events across all dates (numbering restarts each day)")
        
        # Print summary by date
        date_summary = self.significant_df.groupby(self.significant_df['peak_time'].dt.date)['event_id'].nunique()
        for date, count in date_summary.items():
            print(f"[INFO] {date.strftime('%Y-%m-%d')}: {count} events")

    def plot_event_overview(self):
        """
        Create overview plots showing all events on the full time series.
        
        This method generates separate overview plots for each date in the dataset,
        showing the complete time series for that date with event boundaries
        marked as vertical lines. This provides a high-level view of when
        wave events occurred throughout each day.
        
        Features:
        - Separate plot for each date
        - Red dashed lines mark event start times
        - Green dashed lines mark event end times
        - Plots saved as 'full_series_events.png' in each date folder
        
        Side effects:
            - Creates PNG files in date-specific plot folders
            - Displays matplotlib figures temporarily
        """
        print("Creating overview plots by date...")
        events = self.significant_df.groupby("event_id")["peak_time"].agg(["min", "max"]).reset_index()
        events.rename(columns={"min": "start_time", "max": "end_time"}, inplace=True)

        # Group events by date and create separate overview plots for each date
        events_by_date = {}
        for _, event in events.iterrows():
            # Extract date from the event_id (format: YYYYMMDD_XX)
            event_date = event["event_id"].split("_")[0]
            if event_date not in events_by_date:
                events_by_date[event_date] = []
            events_by_date[event_date].append(event)

        # Create overview plot for each date
        for date_str, date_events in events_by_date.items():
            # Get data for this specific date
            date_obj = pd.to_datetime(date_str).date()
            df_date = self.df[self.df.index.date == date_obj]
            
            plt.figure(figsize=(15, 6))
            plt.plot(df_date.index, df_date[self.vertical], label="Vertical Wave", color="blue")
            plt.plot(df_date.index, df_date[self.lateral], label="Lateral Wave", color="orange")

            # Mark event boundaries
            for i, event in enumerate(date_events):
                plt.axvline(event["start_time"], color="red", linestyle="--", alpha=0.8, 
                        label="Start" if i == 0 else "")
                plt.axvline(event["end_time"], color="green", linestyle="--", alpha=0.8, 
                        label="End" if i == 0 else "")

            plt.title(f"Time Series with Event Boundaries - {date_str}")
            plt.xlabel("Time")
            plt.ylabel("Amplitude (m)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.date_folders[date_str]['plots'], "full_series_events.png"))
            plt.close()
        
        print(f"[OK] Overview plots saved for {len(events_by_date)} dates")

    def plot_all_events(self):
        """
        Create detailed plots for each individual wave event.
        
        This method generates a separate detailed plot for each identified wave event,
        showing the wave amplitude in a time window around the event with padding.
        Each plot displays:
        - Complete wave amplitude time series in the window
        - All detected peaks (red dots) and bottoms (green dots)
        - Significant peaks (red circles) and bottoms (green circles) 
        - Event-specific title with date information
        - Event statistics including frequency based on significant peaks
        
        Plots are saved in date-specific folders with format: {date}_{time}_event_{eventnumber}.png
        
        Side effects:
            - Creates numbered PNG files for each event in date folders
            - Shows progress bar during plot generation
        """
        print("Creating individual event plots...")
        events = list(self.significant_df.groupby("event_id"))
        
        if not events:
            print("[WARNING] No events to plot")
            return
        
        with tqdm(total=len(events), desc="Creating plots", unit="plot") as pbar:
            for event_id, group in events:
                # Define time window around event with padding
                start = group["peak_time"].min() - self.padding
                end = group["peak_time"].max() + self.padding
                df_win = self.df[(self.df.index >= start) & (self.df.index <= end)]

                # Extract date and event number from event_id (format: YYYYMMDD_XX)
                date_str, event_num = event_id.split("_")
                output_folder = self.date_folders[date_str]['plots']

                # Get event start time for filename
                event_start_time = group["peak_time"].min()
                time_str = event_start_time.strftime("%H%M%S")  # Format: HHMMSS
                
                # Create filename with new format: {date}_{time}_event_{eventnumber}.png
                filename = f"{date_str}_{time_str}_event_{event_num}.png"

                # Create the plot
                plt.figure(figsize=(14, 6))
                # Calculate mean of vertical signal for the event window and offset
                mean_vertical = df_win[self.vertical].mean()
                df_win_offset = df_win.copy()
                df_win_offset[self.vertical] = df_win[self.vertical] - mean_vertical

                # Plot vertical signal offset to mean = 0
                plt.plot(df_win_offset.index, df_win_offset[self.vertical], label="Vertical Wave (offset)", color="blue")

                # Offset lateral if available
                if self.lateral in df_win.columns:
                    mean_lateral = df_win[self.lateral].mean()
                    df_win_offset[self.lateral] = df_win[self.lateral] - mean_lateral
                    plt.plot(df_win_offset.index, df_win_offset[self.lateral], label="Lateral Wave (offset)", color="orange", linewidth = 0.7)

                # Add horizontal zero line
                plt.axhline(0, color="grey", linestyle="--", linewidth=1)

                # Plot significant peaks and bottoms (larger circles)
                plt.plot(group["peak_time"], group["peak_val"] - mean_vertical, "ro", label="Significant Peaks")
                plt.plot(group["bottom_time"], group["bottom_val"] - mean_vertical, "go", label="Significant Bottoms")

                # === Compute and show event stats ===
                amplitude = group["amplitude_diff"].max()
                num_sig_peaks = group["peak_time"].nunique()
                
                # Calculate event frequency based on significant peaks
                event_start_actual = group["peak_time"].min()
                event_end_actual = group["peak_time"].max()
                
                # If only one significant peak, use a small default duration to avoid division by zero
                if num_sig_peaks == 1:
                    frequency_hz = 0.0  # Single peak has no frequency
                    frequency_text = "N/A (single peak)"
                else:
                    # Calculate duration in seconds
                    event_duration_sec = (event_end_actual - event_start_actual).total_seconds()
                    if event_duration_sec > 0:
                        # Frequency = (number of peaks) / durations
                        frequency_hz = (num_sig_peaks) / event_duration_sec
                        frequency_text = f"{frequency_hz:.2f} Hz"
                    else:
                        frequency_hz = 0.0
                        frequency_text = "N/A (zero duration)"

                # Add stats inside the plot (top-left)
                plt.text(
                    0.01, 0.98,
                    f"Vertical Wave\nAmplitude: {amplitude*1000:.0f} mm\nSign. Peaks: {num_sig_peaks}\nFrequency: {frequency_text}",
                    transform=plt.gca().transAxes,
                    fontsize=10,
                    color = "blue",
                    verticalalignment='top',
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none')
                )

                # Format date and time for display in title
                display_date = event_start_time.strftime("%Y.%m.%d")
                display_time = event_start_time.strftime("%H:%M:%S")
                plt.title(f"Wave Event #{event_num} - {display_date} / {display_time}")
                plt.xlabel("Time")
                plt.ylabel("Amplitude (m)")
                plt.legend(loc="upper right")
                plt.tight_layout()
                plt.savefig(os.path.join(output_folder, filename))
                plt.close("all")
                
                pbar.update(1)
        
        print(f"[OK] Created {len(events)} event plots")

    def save_event_summary_table(self, filename="event_summary.csv"):
        """
        Save summary statistics for all events to CSV files organized by date.
        
        This method creates comprehensive summary tables containing key statistics
        for each wave event, organized into separate CSV files for each date.
        
        Summary statistics include:
        - Event ID and timing information
        - Maximum amplitude difference within the event
        - Count of significant peaks in the event
        - Count of all peaks within the event time range
        - Plot filename for the corresponding event visualization
        
        Args:
            filename (str, optional): Name for the CSV files. Defaults to "event_summary.csv"
            
        Side effects:
            - Creates CSV files in date-specific table folders
            - Each CSV contains events for one date only
        """
        print("Saving event summary tables by date...")
        rows_by_date = {}

        # Process each event and organize by date
        for event_id, group in self.significant_df.groupby("event_id"):
            start_time = group["peak_time"].min()
            end_time = group["peak_time"].max()
            max_amplitude = group["amplitude_diff"].max()
            sig_peaks = group["peak_time"].nunique()
            
            # Count all peaks within the event time range
            all_peaks = self.df.loc[start_time:end_time]["is_peak"].sum()

            # Extract date and event number from event_id (format: YYYYMMDD_XX)
            date_str, event_num = event_id.split("_")
            
            # Generate plot filename using the same format as in plot_all_events()
            event_start_time = group["peak_time"].min()
            time_str = event_start_time.strftime("%H%M%S")  # Format: HHMMSS
            plot_filename = f"{date_str}_{time_str}_event_{event_num}.png"
            
            if date_str not in rows_by_date:
                rows_by_date[date_str] = []

            # Build summary row for this event
            rows_by_date[date_str].append({
                "event_id": int(event_num),  # Convert to integer for cleaner display
                "event_start": start_time,
                "event_end": end_time,
                "amplitude": round(max_amplitude,3),
                "number_significant_peaks": sig_peaks,
                "number_all_peaks": all_peaks,
                "plot_file_name": plot_filename
            })

        # Save separate CSV files for each date
        for date_str, rows in rows_by_date.items():
            summary_df = pd.DataFrame(rows)
            # Sort by event_id to ensure proper order
            summary_df = summary_df.sort_values("event_id").reset_index(drop=True)
            summary_path = os.path.join(self.date_folders[date_str]['tables'], filename)
            summary_df.to_csv(summary_path, index=False)
        
        print(f"[OK] Event summary tables saved for {len(rows_by_date)} dates")

    def run_all(self):
        """
        Execute the complete wave analysis pipeline.
        
        This method runs the entire analysis workflow in the correct sequence:
        1. Load and preprocess the wave data using DataLoader
        2. Detect peaks and bottoms in the signal
        3. Find significant peak-bottom pairs above threshold
        4. Group pairs into discrete wave events
        5. Generate overview plots for each date
        6. Create detailed plots for each event
        7. Save summary statistics tables
        
        The method includes comprehensive error handling and provides status
        updates throughout the process.
        
        Raises:
            Exception: Re-raises any exception that occurs during analysis
            
        Side effects:
            - Creates complete analysis output in organized date folders
            - Prints progress and completion status
        """
        print("Starting Wave Event Analysis")
        print("=" * 50)
        
        try:
            self.load_data()
            self.detect_peaks_and_bottoms()
            self.find_significant_pairs()
            self.group_events()
            self.plot_event_overview()
            self.plot_all_events()
            self.save_event_summary_table()
            
            print("=" * 50)
            print("Analysis completed successfully!")
            print(f"Results organized by date in:")
            print(f"Plots: {self.output_base_dir}")
            print(f"Tables: {self.tables_base_dir}")
            print(f"Date folders created: {list(self.date_folders.keys())}")
            
        except Exception as e:
            print(f"[ERROR] Error during analysis: {str(e)}")
            raise


if __name__ == "__main__":
    """
    Main execution block for command-line usage.
    
    This block handles command-line arguments, validates input files,
    and runs the complete analysis pipeline.
    
    Usage:
        python analyzer.py [config.yaml]
        
    If no config file is specified, defaults to 'config.yaml' in the current directory.
    """
    # Check and install required packages first
    check_and_install_packages()
    
    # Parse command line arguments
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    
    # Validate configuration file exists
    if not os.path.exists(config_path):
        print(f"[ERROR] Config file not found: {config_path}")
        print("Usage: python analyzer.py [config.yaml]")
        sys.exit(1)
    
    # Run the analysis
    analyzer = WaveEventAnalyzer(config_path)
    analyzer.run_all()