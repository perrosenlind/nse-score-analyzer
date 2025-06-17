# NSE Score Analyzer

A Python tool to analyze PDF score reports and extract exam information, candidate details, and performance metrics with visual progress bars.

## Features

- **PDF Analysis**: Extract exam information, candidate name, grade, and date from PDF score reports
- **Topic Extraction**: Automatically identify and extract topic areas from the report
- **Bar Chart Analysis**: Analyze progress bars in the PDF to determine performance percentages
- **Rich Console Output**: Beautiful, colorful terminal output with icons and formatted tables
- **Error Handling**: Comprehensive error checking for missing or invalid PDF files
- **Debug Mode**: Saves intermediate images for troubleshooting bar detection

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd nse-score-analyzer
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Install Tesseract OCR:
   - **macOS**: `brew install tesseract`
   - **Ubuntu/Debian**: `sudo apt-get install tesseract-ocr`
   - **Windows**: Download from [GitHub releases](https://github.com/UB-Mannheim/tesseract/wiki)

## Usage

### Basic Usage

Analyze a PDF file named `scorereport.pdf` in the current directory:
```bash
python nse-score-analyzer.py
```

### Specify a Different PDF File

```bash
python nse-score-analyzer.py -f /path/to/your/report.pdf
```

or

```bash
python nse-score-analyzer.py --file /path/to/your/report.pdf
```

### Example Output

```
Exam Name: Network Security Essentials
Candidate Name: John Doe
✅ Pass
Overall score: 78.4 %
Date: 2024-01-15

┏━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Topic                  ┃ Score ┃ Bar                            ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Network Fundamentals   │ 85%   │ |█████████████████████████     │
│ Security Protocols     │ 72%   │ |█████████████████████         │
│ Threat Analysis        │ 90%   │ |███████████████████████████   │
│ Risk Management        │ 68%   │ |████████████████████          │
│ Incident Response      │ 77%   │ |███████████████████████       │
└────────────────────────┴───────┴────────────────────────────────┘
```

## Error Handling

The tool provides clear error messages for common issues:

- **File not found**: Shows the current working directory and usage instructions
- **Invalid PDF**: Warns if the file doesn't have a .pdf extension
- **Corrupted PDF**: Catches and reports PDF parsing errors
- **No topics found**: Gracefully handles reports without detectable topics

## Technical Details

### Dependencies

- **PyMuPDF (fitz)**: PDF parsing and rendering
- **Pillow (PIL)**: Image processing and manipulation
- **OpenCV (cv2)**: Computer vision for bar chart analysis
- **NumPy**: Numerical operations for image analysis
- **pytesseract**: OCR for text extraction from images
- **Rich**: Beautiful terminal output with colors and tables

### How It Works

1. **PDF Parsing**: Uses PyMuPDF to extract text and render pages as images
2. **Text Extraction**: Searches for specific patterns to identify exam information
3. **Image Analysis**: Converts PDF pages to images for bar chart detection
4. **Bar Detection**: Uses OpenCV to identify and analyze progress bars
5. **Percentage Calculation**: Analyzes filled portions of bars to calculate scores
6. **Output Formatting**: Uses Rich library for formatted console output

### Debug Files

The tool generates debug images for troubleshooting:
- `debug_bar_area.png`: Shows the detected bar area
- `debug_bars_detected.png`: Highlights detected bars with green rectangles
- `debug_left_crop.png`: Shows the cropped area used for OCR

## Troubleshooting

### Common Issues

1. **No bars detected**: The PDF might have a different layout. Check debug images.
2. **Incorrect topic extraction**: OCR might struggle with image quality. Try with a higher resolution PDF.
3. **Missing dependencies**: Ensure all requirements are installed and Tesseract is in your PATH.

### Debug Mode

Debug images are automatically saved to help troubleshoot detection issues. Examine these files if the tool isn't detecting bars or topics correctly.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license information here]

## Support

If you encounter issues or have questions, please open an issue on the GitHub repository.