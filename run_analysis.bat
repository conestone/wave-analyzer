@echo off
REM Wave Analysis Batch Script
REM This script runs the wave event analyzer with the specified config file
REM Change to the directory containing the script
cd /d "%~dp0"

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python and try again
    pause
    exit /b 1
)

REM Check if analyzer.py exists
if not exist "code\analyzer.py" (
    echo [ERROR] analyzer.py not found in code\ directory
    echo Please make sure the file exists and try again
    pause
    exit /b 1
)

REM Check if config.yaml exists
if not exist "config.yaml" (
    echo [ERROR] config.yaml not found in current directory
    echo Please make sure the config file exists and try again
    pause
    exit /b 1
)

echo [INFO] Running wave analysis...
echo [INFO] Script: code\analyzer.py
echo [INFO] Config: config.yaml
echo.

REM Run the Python script with the config file
python code\analyzer.py config.yaml

REM Check if the script executed successfully
if %errorlevel% equ 0 (
    echo.
    echo =====================================
    echo Analysis completed successfully!
    echo =====================================
) else (
    echo.
    echo =====================================
    echo Analysis failed with error code: %errorlevel%
    echo =====================================
)

echo.
echo Press any key to exit...
pause >nul