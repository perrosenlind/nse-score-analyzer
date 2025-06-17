import fitz  # PyMuPDF
from PIL import Image, ImageEnhance
import numpy as np
import cv2
import io
from rich.console import Console
from rich.table import Table
from rich.text import Text
import argparse
import pytesseract
import os
import sys


def extract_images_from_pdf(pdf_path):
    """Extract images from PDF file."""
    doc = fitz.open(pdf_path)
    images = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        for img_idx, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            pil_img = Image.open(io.BytesIO(image_bytes))
            images.append(pil_img)
            # Save for inspection
            out_path = f"extracted_image_page{page_num + 1}_{img_idx}.png"
            pil_img.save(out_path)
            print(f"Saved {out_path} size={pil_img.size}")
    doc.close()
    return images


def extract_labels(image, num_labels=5):
    """Extract labels from image using OCR."""
    # Crop a wider left side
    left_crop = image.crop((0, 0, int(image.width * 0.65), image.height))
    # Save for debugging
    left_crop.save("debug_left_crop.png")
    # Enhance for better OCR
    left_crop = left_crop.convert('L')
    enhancer = ImageEnhance.Contrast(left_crop)
    left_crop = enhancer.enhance(2.5)
    # OCR with different PSM
    text = pytesseract.image_to_string(left_crop, config='--psm 6')
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    print("OCR lines:", lines)
    for line in lines[:num_labels]:
        print(line)
    return lines[:num_labels]


def extract_text_rows(pdf_path):
    """Extract text rows from PDF."""
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text() # type: ignore
        print(f"--- Page {page_num + 1} ---")
        print(text)
    doc.close()


def extract_exam_info_and_topics(pdf_path):
    """Extract exam information and topics from PDF."""
    doc = fitz.open(pdf_path)
    exam_name = candidate_name = grade = date = None
    topics = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text() # type: ignore
        lines = text.splitlines()

        # Exam name (row after "XAMINATION SUMMARY REPORT")
        for i, line in enumerate(lines):
            if "XAMINATION SUMMARY REPORT" in line.upper() and i + 1 < len(lines):
                exam_name = lines[i + 1].strip()
                break

        # Candidate info (row after each label)
        for i, line in enumerate(lines):
            if "Candidate Name:" in line and i + 1 < len(lines):
                candidate_name = lines[i + 1].strip()
            if "Grade:" in line and i + 1 < len(lines):
                grade = lines[i + 1].strip()
            if "Date:" in line and i + 1 < len(lines):
                date = lines[i + 1].strip()

        # Find the line ending with "topic."
        start = None
        for i, line in enumerate(lines):
            if line.strip().endswith("topic."):
                start = i + 1
                break

        # Find the next line ending with "."
        end = None
        if start is not None:
            for i in range(start, len(lines)):
                if lines[i].strip().endswith("."):
                    end = i
                    break

        # Extract topics
        if start is not None and end is not None:
            topics = [line.strip() for line in lines[start:end] if line.strip()]
        else:
            print("DEBUG: Could not find topic markers. Printing all lines for inspection:")
            for idx, line in enumerate(lines):
                print(f"{idx}: {line}")
        break  # Only process first page
    doc.close()
    return exam_name, candidate_name, grade, date, topics


def extract_bar_percentages(pdf_path, num_bars):
    """Extract bar percentages from PDF images."""
    doc = fitz.open(pdf_path)
    images = []
    for page in doc:
        for img in page.get_images(full=True):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            images.append(Image.open(io.BytesIO(image_bytes)))
    doc.close()
    
    if not images:
        return []
    # Assume first image is the bar chart
    image = images[0].convert('L')
    img_np = np.array(image)
    height, width = img_np.shape
    # Skip the top 60% of the image
    bar_area = img_np[int(height*0.9):, :]
    # Threshold
    _, thresh = cv2.threshold(bar_area, 200, 255, cv2.THRESH_BINARY_INV)
    # Save debug threshold image
    Image.fromarray(thresh).save("debug_bar_area.png")
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bars = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Heuristic: bars are wide and not too thin
        if w > bar_area.shape[1] * 0.2 and h < bar_area.shape[0] // num_bars * 1.5:
            bars.append((x, y, w, h))
    if not bars:
        return []
    # Debug: draw rectangles
    debug_img = cv2.cvtColor(bar_area, cv2.COLOR_GRAY2BGR)
    for x, y, w, h in bars:
        cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imwrite("debug_bars_detected.png", debug_img)
    # Sort bars top-to-bottom
    bars = sorted(bars, key=lambda b: b[1])
    bars = bars[:num_bars]
    max_width = max([w for (_, _, w, _) in bars])
    percentages = [round((w / max_width) * 100, 2) for (_, _, w, _) in bars]
    return percentages


def extract_bar_percentages_from_image(image, num_bars):
    """Extract bar percentages from image."""
    img_np = np.array(image.convert('L'))
    # Crop to the likely bar area (adjust as needed)
    height, width = img_np.shape
    bar_area = img_np[int(height*0.35):int(height*0.95), int(width*0.45):]
    _, thresh = cv2.threshold(bar_area, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bars = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 50 and h < 40 and w/h > 4:
            bars.append((x, y, w, h))
    # Debug: draw rectangles
    debug_img = cv2.cvtColor(bar_area, cv2.COLOR_GRAY2BGR)
    for x, y, w, h in bars:
        cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imwrite("debug_bars_detected.png", debug_img)
    # Sort and get percentages
    bars = sorted(bars, key=lambda b: b[1])
    bars = bars[:num_bars]
    max_width = max([w for (_, _, w, _) in bars]) if bars else 1
    percentages = [round((w / max_width) * 100, 2) for (_, _, w, _) in bars]
    return percentages


def render_page_as_image(pdf_path, page_num=0, zoom=2):
    """Render PDF page as image."""
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat) # type: ignore
    img = Image.open(io.BytesIO(pix.tobytes()))
    doc.close()
    return img


def extract_bar_percentages_from_rendered(image, num_bars):
    """Extract bar percentages from rendered image."""
    img_np = np.array(image.convert('L'))
    height, width = img_np.shape
    # Skip the top 60% (adjust as needed)
    bar_area = img_np[int(height*0.6):, :]
    _, thresh = cv2.threshold(bar_area, 200, 255, cv2.THRESH_BINARY_INV)
    # Save debug threshold image
    Image.fromarray(thresh).save("debug_bar_area.png")
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bars = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > bar_area.shape[1] * 0.2 and h < bar_area.shape[0] // num_bars * 1.5:
            bars.append((x, y, w, h))
    # Debug: draw rectangles
    debug_img = cv2.cvtColor(bar_area, cv2.COLOR_GRAY2BGR)
    for x, y, w, h in bars:
        cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imwrite("debug_bars_detected.png", debug_img)
    bars = sorted(bars, key=lambda b: b[1])
    bars = bars[:num_bars]
    max_width = max([w for (_, _, w, _) in bars]) if bars else 1
    percentages = [round((w / max_width) * 100, 2) for (_, _, w, _) in bars]
    return percentages


def extract_bar_fill_percentages_from_rendered(image, num_bars):
    """Extract bar fill percentages from rendered image."""
    img_np = np.array(image.convert('L'))
    height, width = img_np.shape
    # Skip the top 60% (adjust as needed)
    bar_area = img_np[int(height*0.6):, :]
    _, thresh = cv2.threshold(bar_area, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bars = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > bar_area.shape[1] * 0.2 and h < bar_area.shape[0] // num_bars * 1.5:
            bars.append((x, y, w, h))
    bars = sorted(bars, key=lambda b: b[1])
    bars = bars[:num_bars]
    percentages = []
    for x, y, w, h in bars:
        # Use the original grayscale bar area for fill detection
        bar_crop = bar_area[y:y+h, x:x+w]
        # Use a lower threshold to detect the filled (dark) part
        fill_thresh = 100  # adjust if needed
        filled = np.mean(bar_crop, axis=0) < fill_thresh  # True for filled columns
        if np.any(filled):
            filled_width = np.argmax(~filled)  # first unfilled column from left
            if filled_width == 0:
                filled_width = np.sum(filled)  # all filled
        else:
            filled_width = 0
        percent = 100 * filled_width / w
        percentages.append(round(percent, 2))
    return percentages


def mark_bar_filling_on_image(image, num_bars=5):
    """Mark bar filling on image and return percentages."""
    img_np = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    crop_top = int(height * 0.4)  # Was 0.6, now 0.4 to include more at the top
    bar_area = gray[crop_top:, :]
    _, thresh = cv2.threshold(bar_area, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bars = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw > bar_area.shape[1] * 0.2 and bh < bar_area.shape[0] // num_bars * 1.5:
            bars.append((x, y, bw, bh))
    bars = sorted(bars, key=lambda b: b[1])
    bars = bars[:num_bars]
    percentages = []
    for x, y, bw, bh in bars:
        bar_crop = bar_area[y:y+bh, x:x+bw]
        col_means = np.mean(bar_crop, axis=0)
        fill_thresh = 80  # Adjust if needed
        filled_cols = np.where(col_means < fill_thresh)[0]
        if len(filled_cols) > 0:
            filled_width = filled_cols[-1] + 1
            percent = round((filled_width / bw) * 100, 2)
        else:
            filled_width = 0
            percent = 0.0
        percentages.append(percent)
    return percentages


def main():
    """Main function to analyze PDF score report."""
    parser = argparse.ArgumentParser(description="Analyze a score report PDF.")
    parser.add_argument(
        "-f", "--file",
        type=str,
        default="scorereport.pdf",
        help="Path to the PDF file to analyze (default: scorereport.pdf)"
    )
    args = parser.parse_args()
    pdf_path = args.file

    # Check if PDF file exists
    if not os.path.exists(pdf_path):
        console = Console()
        error_text = Text(f"❌ Error: PDF file '{pdf_path}' not found!", style="bold red")
        console.print(error_text)
        console.print("Please check that the file path is correct and the file exists.")
        console.print(f"Current working directory: {os.getcwd()}")
        console.print("Use -f or --file to specify a different PDF file path.")
        sys.exit(1)

    # Check if file is actually a PDF
    if not pdf_path.lower().endswith('.pdf'):
        console = Console()
        warning_text = Text(f"⚠️  Warning: '{pdf_path}' does not appear to be a PDF file!", style="bold yellow")
        console.print(warning_text)

    # Try to open the PDF file
    try:
        test_doc = fitz.open(pdf_path)
        test_doc.close()
    except Exception as e:
        console = Console()
        error_text = Text(f"❌ Error: Cannot open PDF file '{pdf_path}'", style="bold red")
        console.print(error_text)
        console.print(f"Error details: {str(e)}")
        console.print("Please ensure the file is a valid PDF and not corrupted.")
        sys.exit(1)

    exam_name, candidate_name, grade, date, topics = extract_exam_info_and_topics(pdf_path)
    print("\nExam Name:", exam_name)
    print("Candidate Name:", candidate_name)

    # Color grade and add icon
    grade_icon = "❌"
    grade_color = "bold red"
    if grade and "pass" in grade.lower():
        grade_icon = "✅"
        grade_color = "bold green"
    grade_text = Text(f"{grade_icon} {grade}", style=grade_color)
    console = Console()
    console.print("Grade:", grade_text)

    # Print overall score as before
    if topics:
        rendered_img = render_page_as_image(pdf_path, page_num=0, zoom=2)
        bar_percentages = mark_bar_filling_on_image(rendered_img, num_bars=len(topics))
        overall_score = round(sum(bar_percentages) / len(bar_percentages), 2)
        print(f"Overall score: {overall_score} %")
    else:
        print("Overall score: N/A")

    print("Date:", date, "\n")

    if not topics:
        print("No topics found in the document.")
    else:
        console = Console()
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Topic")
        table.add_column("Score", justify="right")
        table.add_column("Bar")

        bar_length = 30  # Adjust bar length as desired

        for idx, topic in enumerate(topics):
            percent = bar_percentages[idx] if idx < len(bar_percentages) else 0
            percent_str = f"{percent}%"
            filled_length = int(bar_length * percent / 100)
            bar_chars = ["█"] * filled_length + [" "] * (bar_length - filled_length)
            bar_body = "".join(bar_chars)
            bar_text = Text(f"|{bar_body}|")
            table.add_row(topic, percent_str, bar_text)

        console.print(table)


if __name__ == "__main__":
    main()