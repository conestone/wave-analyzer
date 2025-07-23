# Wave Event Analyzer

A comprehensive Python tool for analyzing wave events from boat wave time series data. This analyzer identifies significant wave peaks and bottoms, groups them into events, and generates detailed visualizations and summary reports organized by date.

## Features

- **Multi-File Processing**: Processes multiple CSV files from input directory automatically with dynamic sample column detection
- **Automated Peak Detection**: Uses scipy signal processing to identify wave peaks and bottoms
- **Event Grouping**: Groups nearby significant wave events based on configurable time gaps
- **Date Organization**: Automatically organizes outputs into date-specific folders (YYYYMMDD format)
- **Comprehensive Visualization**: 
  - Overview plots showing all events on the full time series for each date
  - Detailed individual event plots with peak/bottom markers and comprehensive statistics
  - Automatic mean offset for better visualization
  - Dual-wave analysis (vertical and lateral) with color-coded statistics
- **Enhanced Statistics**: Detailed deflection analysis and amplitude measurements for both wave components
- **Summary Reports**: Comprehensive CSV tables with event statistics organized by date
- **Flexible Input Support**: Supports variable number of sample columns (not limited to 32)
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
vertical_variable: "F001"        # Name of vertical wave column
lateral_variable: "F002"         # Name of lateral wave column

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
| `vertical_variable` | Vertical wave amplitude column name | "F001" | Column name |
| `lateral_variable` | Lateral wave amplitude column name | "F002" | Column name |
| `sampling_rate_hz` | Data sampling frequency | 8 | Hz |
| `amplitude_threshold` | Minimum significant wave height | 0.05 | meters |
| `significant_window_sec` | Peak search window | 5 | seconds |
| `event_gap_sec` | Event separation threshold | 60 | seconds |
| `plot_padding_sec` | Plot time padding | 60 | seconds |

## Input Data Format

### Method 1: Multiple CSV Files (Recommended)
Place multiple CSV files in the `input/` directory. Each file should contain headerless data with:
- **Column 1**: Direction identifier (e.g., F002, A003)
- **Column 2**: s parameter
- **Column 3**: b parameter  
- **Column 4**: Date/timestamp
- **Columns 5+**: Sample data (sample1, sample2, ..., sampleN)

**Key Features:**
- **Dynamic Sample Detection**: Automatically detects any number of sample columns (not limited to 32)
- **No Headers Required**: Files are processed without headers
- **Flexible Sample Count**: Supports varying numbers of samples per file

Example CSV structure (no headers):
```csv
F002,0,0,2025-01-15 10:00:00,0.123,0.156,0.189,0.145,...,0.089
A003,0,0,2025-01-15 10:00:00,0.045,0.067,0.078,0.056,...,0.034
F002,0,0,2025-01-15 10:00:01,0.234,0.267,0.298,0.276,...,0.198
...
```

### Method 2: Legacy Single CSV File
Single CSV file with time series data containing:
- **Time column**: Either `zeitstempel` (German) or `timestamp` (English)
  - Format: `YYYY-MM-DD HH:MM:SS.ffffff`
- **Wave amplitude columns**: Specified in config parameters

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
- **Vertical Wave Statistics (Blue Text Box)**:
  - Amplitude (mm)
  - Max Deflection (peak distance from mean, mm)
  - Min Deflection (bottom distance from mean, mm)
  - Number of significant peaks
  - Event duration

- **Lateral Wave Statistics (Orange Text Box)**:
  - Amplitude (mm)
  - Max Deflection (peak distance from mean, mm)
  - Min Deflection (bottom distance from mean, mm)

- **Visual Elements**:
  - Both wave signals offset to mean = 0 for better comparison
  - Significant peaks (red circles) and bottoms (green circles)
  - Zero reference line
  - Color-coded statistics matching wave line colors

### Tables (output/tables/YYYYMMDD/)
- **`event_summary.csv`**: Comprehensive summary statistics for all events on that date

### Event Summary Table Columns
| Column | Description |
|--------|-------------|
| `event_id` | Sequential event identifier (resets each day) |
| `event_start` | Event start timestamp |
| `event_end` | Event end timestamp |
| `amplitude` | Maximum amplitude difference in event (meters) |
| `number_significant_peaks` | Count of significant peaks |
| `number_all_peaks` | Total peaks in event time range |
| `mean_vertical` | Mean vertical wave value during event |
| `vertical_max_deflection` | Maximum vertical deflection from mean |
| `vertical_min_deflection` | Minimum vertical deflection from mean |
| `mean_lateral` | Mean lateral wave value during event (if available) |
| `lateral_max_deflection` | Maximum lateral deflection from mean (if available) |
| `lateral_min_deflection` | Minimum lateral deflection from mean (if available) |
| `plot_file_name` | Corresponding plot filename |

## Algorithm Overview

1. **Dynamic Data Loading**: 
   - DataLoader automatically detects number of sample columns
   - Processes any number of samples per row
   - Handles multiple CSV files with different sample counts
   - Falls back to legacy single CSV if needed

2. **Peak Detection**: Use scipy.signal.find_peaks to identify local maxima and minima

3. **Significance Filtering**: Find peak-bottom pairs exceeding amplitude threshold within time window

4. **Event Grouping**: Group nearby significant pairs into discrete events based on time gaps

5. **Dual-Wave Analysis**: Calculate comprehensive statistics for both vertical and lateral components

6. **Deflection Analysis**: Measure peak and bottom distances from respective mean lines

7. **Date Organization**: Automatically create date-specific output folders

8. **Enhanced Visualization**: Generate color-coded plots with comprehensive statistics

9. **Comprehensive Reporting**: Create detailed summary tables with all calculated metrics

## Key Improvements in Version 1.2

- **Dynamic Sample Column Detection**: Automatically handles any number of sample columns
- **Enhanced Deflection Analysis**: Max/min deflection calculations for both wave components
- **Improved Statistics Display**: Color-coded text boxes with comprehensive wave metrics
- **Better Table Output**: Extended CSV tables with deflection data and mean values
- **Robust Data Handling**: Improved duplicate detection and error handling
- **Flexible Input Support**: No longer limited to 32 samples per row
- **Duration Calculation**: Event duration replaces frequency calculation for better clarity

## Deflection Analysis

The analyzer now provides detailed deflection analysis:

- **Max Deflection**: Largest distance from any peak to the mean line
- **Min Deflection**: Largest distance from any bottom to the mean line
- **Mean Offset**: All plots show signals offset to their respective means for better comparison
- **Dual Analysis**: Separate deflection calculations for vertical and lateral waves

This provides insight into:
- Wave symmetry around the mean
- Maximum excursions in both directions
- Relative amplitudes between wave components

## Troubleshooting

### Common Issues

**"No CSV files found in input directory"**
- Ensure CSV files exist in the `input/` folder
- Check that files have `.csv` extension

**"Missing required columns"**
- Verify `vertical_variable` and `lateral_variable` in config match your data direction identifiers
- Check column names are exact matches (case-sensitive)

**"No sample columns detected"**
- Ensure CSV files have at least 5 columns (direction, s, b, date + samples)
- Verify file structure matches expected format

**"Index contains duplicate entries"**
- The analyzer now automatically handles duplicates
- Check data quality if frequent duplicates occur

**"Config file not found"**
- Verify `config.yaml` exists in the project root
- Check file path in batch script or command line

**Empty output**
- Check if `amplitude_threshold` is too high for your data
- Verify direction identifiers match your data exactly
- Check if sampling rate is correct
- Ensure data contains actual wave measurements

### Performance Notes

- CSV files with any number of sample columns are processed efficiently
- Progress bars show processing status for large datasets
- Date-specific organization improves output management
- Automatic duplicate handling prevents pivot errors
- Consider adjusting `event_gap_sec` and `significant_window_sec` for optimal event detection

### Data Quality

- The analyzer handles variable sample counts automatically
- Duplicate timestamps are detected and resolved
- Missing values are properly handled
- Data validation ensures robust processing

## Author

**Oliver Konold**  
Version: 1.2 - Enhanced Deflection Analysis and Dynamic Sample Detection  
Year: 2025
