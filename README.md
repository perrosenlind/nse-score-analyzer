# NSE Score Analyzer

A Python tool that extracts and analyzes Fortinet NSE certification exam score reports from PDF files.

## Overview

This tool helps in extracting and visualizing exam details from Fortinet NSE certification score reports, including:

- Exam name and candidate information
- Overall score and pass/fail status
- Individual topic scores with visual bars
- Detailed score breakdown

## Features

- PDF image and text extraction
- Automated bar chart score detection and analysis
- Rich console output with colorful formatting
- Visual progress bars for topic scores
- Command-line interface with file selection

## Requirements

- Python 3.6+
- Required libraries:
  - PyMuPDF (fitz)
  - PIL (Pillow)
  - numpy
  - opencv-python (cv2)
  - rich
  - argparse

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/nse-score-analyzer.git
   cd nse-score-analyzer
   ```

2. Set up a virtual environment (recommended):
   ```
   python -m venv venv
   
   # On macOS/Linux
   source venv/bin/activate
   
   # On Windows
   venv\Scripts\activate
   ```

3. Install the required dependencies using requirements.txt:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the script with the path to your score report PDF:

```bash
python score-analyze.py -f /path/to/your/scorereport.pdf
```

By default, it will look for a file named `scorereport.pdf` in the current directory.

## Output

The tool provides a rich console output with:
- Exam name and candidate details
- Color-coded pass/fail status
- Overall score percentage
- Table of topics with score percentages and visual bars

## Notes

- The script creates debug images during analysis that can be helpful for troubleshooting
- Bar detection parameters may need adjustment for different PDF formats