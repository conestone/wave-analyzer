# Wave Event Analyzer

A comprehensive Python tool for analyzing wave events from boat wave time series data. This analyzer identifies significant wave peaks and bottoms, groups them into events, and generates detailed visualizations and summary reports organized by date.

## Features

- **Multi-File Processing**: Processes multiple CSV files from input directory automatically
- **Automated Peak Detection**: Uses scipy signal processing to identify wave peaks and bottoms
- **Event Grouping**: Groups nearby significant wave events based on configurable time gaps
- **Date Organization**: Automatically organizes outputs into date-specific folders (YYYYMMDD format)
- **Comprehensive Visualization**: 
  - Overview plots showing all events on the full time series for each date
  - Detailed individual event plots with peak/bottom markers and statistics
  - Automatic mean offset for better visualization
- **Summary Reports**: CSV tables with event statistics organized by date
- **Flexible Configuration**: All analysis parameters configurable via YAML file
- **Auto-Dependency Management**: Automatically checks and installs required packages

## Requirements

The script automatically checks and installs required packages:
- Python 3.x
- PyYAML
- pandas
- numpy
- matplotlib
- scipy
- tqdm

## File Structure

```
project/
├── analyzer.py             # Main analysis script
├── input/                  # Place your CSV data files here
│   ├── file1.csv          # Multiple CSV files supported
│   ├── file2.csv
│   └── ...
├── output/
│   ├── plots/             # Generated plots organized by date
│   │   ├── 20250101/      # Date-specific folders (YYYYMMDD)
│   │   │   ├── full_series_events.png
│   │   │   ├── 20250101_103045_event_01.png
│   │   │   └── 20250101_143012_event_02.png
│   │   └── 20250102/
│   └── tables/            # Summary CSV files organized by date
│       ├── 20250101/
│       │   └── event_summary.csv
│       └── 20250102/
├── config.yaml            # Configuration file
├── run_analysis.bat       # Windows batch script to run analysis
└── README.md             # This file
```

## Configuration

Edit `config.yaml` to customize the analysis parameters:

```yaml
# Data specific parameters 
vertical_variable: "F002"        # Name of vertical wave column
lateral_variable: "A003"         # Name of lateral wave column

# Analysis specific parameters
sampling_rate_hz: 8               # Sampling frequency in Hz
amplitude_threshold: 0.05         # Minimum peak-bottom difference (meters)
significant_window_sec: 5         # Time window around peaks (seconds)
event_gap_sec: 60                # Max gap between peaks in same event (seconds)
plot_padding_sec: 60              # Plot padding around events (seconds)
```

### Configuration Parameters

| Parameter | Description | Default | Units |
|-----------|-------------|---------|-------|
| `vertical_variable` | Vertical wave amplitude column name | "F002" | Column name |
| `lateral_variable` | Lateral wave amplitude column name | "A003" | Column name |
| `sampling_rate_hz` | Data sampling frequency | 8 | Hz |
| `amplitude_threshold` | Minimum significant wave height | 0.05 | meters |
| `significant_window_sec` | Peak search window | 5 | seconds |
| `event_gap_sec` | Event separation threshold | 60 | seconds |
| `plot_padding_sec` | Plot time padding | 60 | seconds |

## Input Data Format

### Method 1: Multiple CSV Files (Recommended)
Place multiple CSV files in the `input/` directory. Each file should contain:
- **Direction column**: Wave measurement direction identifier (lateral, vertical)
- **Date column**: Date information for measurements
- **Sample columns**: sample columns (sample1, sample2, ..., sample32) containing wave measurements

### Method 2: Legacy Single CSV File
Single CSV file with time series data containing:
- **Time column**: Either `zeitstempel` (German) or `timestamp` (English)
  - Format: `YYYY-MM-DD HH:MM:SS.ffffff`
- **Wave amplitude columns**: Specified in config parameters

Example multi-file CSV structure:
```csv
direction,s,b,date,sample1,sample2,...,sample32
F002,0,0,2025-01-15 10:00:00,0.123,0.156,...,0.089
A003,0,0,2025-01-15 10:00:00,0.045,0.067,...,0.034
...
```

## Usage

### Windows (Recommended)
1. Double-click `run_analysis.bat`
2. The script will automatically check dependencies and run the analysis

### Command Line
```bash
# Navigate to project directory
cd /path/to/your/project

# Run with default config
python analyzer.py

# Run with custom config file
python analyzer.py custom_config.yaml
```

## Output

### Plots (output/plots/YYYYMMDD/)
- **`full_series_events.png`**: Overview showing all events for the specific date
- **`YYYYMMDD_HHMMSS_event_XX.png`**: Detailed individual event plots with timing information

### Individual Event Plots Include:
- Vertical and lateral wave signals (offset to mean = 0)
- Significant peaks (red circles) and bottoms (green circles)
- Event statistics: amplitude, number of significant peaks, frequency
- Zero reference line for better visualization

### Tables (output/tables/YYYYMMDD/)
- **`event_summary.csv`**: Summary statistics for all events on that date

### Event Summary Table Columns
| Column | Description |
|--------|-------------|
| `event_id` | Sequential event identifier (resets each day) |
| `event_start` | Event start timestamp |
| `event_end` | Event end timestamp |
| `amplitude` | Maximum amplitude difference in event (meters) |
| `number_significant_peaks` | Count of significant peaks |
| `number_all_peaks` | Total peaks in event time range |
| `plot_file_name` | Corresponding plot filename |

## Algorithm Overview

1. **Data Loading**: 
   - Uses DataLoader class to process multiple CSV files
   - Expands 32 samples per row into proper time series
   - Falls back to legacy single CSV if needed
2. **Peak Detection**: Use scipy.signal.find_peaks to identify local maxima and minima
3. **Significance Filtering**: Find peak-bottom pairs exceeding amplitude threshold within time window
4. **Event Grouping**: Group nearby significant pairs into discrete events based on time gaps
5. **Date Organization**: Automatically create date-specific output folders
6. **Visualization**: Generate overview and detailed plots for each date
7. **Reporting**: Create summary statistics organized by date

## Key Improvements in Version 1.1

- **Multi-File Support**: Processes all CSV files in input directory automatically
- **DataLoader Integration**: Proper handling of multi-sample data format
- **Date-Based Organization**: All outputs organized by date (YYYYMMDD folders)
- **Enhanced Plotting**: Better filenames with date/time information
- **Improved Statistics**: Event frequency calculations and enhanced visualizations
- **Auto-Dependency Management**: Automatic package installation
- **Event Numbering**: Events numbered separately for each date (1, 2, 3, etc.)

## Troubleshooting

### Common Issues

**"No CSV files found in input directory"**
- Ensure CSV files exist in the `input/` folder
- Check that files have `.csv` extension

**"Missing required columns"**
- Verify `vertical_variable` and `lateral_variable` in config match your data columns
- Check column names are exact matches (case-sensitive)

**"Config file not found"**
- Verify `config.yaml` exists in the project root
- Check file path in batch script or command line

**"Time column not found"** (Legacy mode)
- Ensure your CSV has either `zeitstempel` or `timestamp` column
- Only applies when falling back to single CSV file mode

**Missing packages**
- The script auto-installs packages, but ensure you have internet connectivity
- For offline use, manually install: `pip install PyYAML pandas numpy matplotlib scipy tqdm`

**Empty output**
- Check if `amplitude_threshold` is too high for your data
- Verify column names match your data exactly
- Check if sampling rate is correct
- Ensure data contains actual wave measurements

### Performance Notes

- Multiple CSV files are processed efficiently with progress bars
- Large datasets may take several minutes to process
- Date-specific organization improves output management
- Consider adjusting `event_gap_sec` and `significant_window_sec` for optimal event detection

### Output Organization

- All outputs are automatically organized by date
- Each date gets separate folders for plots and tables
- Event numbering restarts at 1 for each date
- Filenames include date/time information for easy reference

## Author

**Oliver Konold**  
Version: 1.1 
Year: 2025