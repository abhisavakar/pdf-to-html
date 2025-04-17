import os
import uuid
import time
import re
import io
import concurrent.futures
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, jsonify
import fitz  # PyMuPDF
import pdfminer
from pdfminer.high_level import extract_text as pdfminer_extract_text
from pdfminer.layout import LAParams
import PyPDF2
import tempfile
from pdf2image import convert_from_path
import pytesseract
from werkzeug.utils import secure_filename
from bs4 import BeautifulSoup
import html
import logging
from functools import lru_cache
import numpy as np
from collections import Counter

# Try to import optional libraries
try:
    import camelot

    CAMELOT_AVAILABLE = True
except ImportError:
    CAMELOT_AVAILABLE = False

try:
    from unstructured.partition.pdf import partition_pdf

    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['TEMP_FOLDER'] = 'temp'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.secret_key = os.environ.get('SECRET_KEY', 'supersecretkey')

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['TEMP_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'pdf'}

# Maximum worker threads for parallel processing
MAX_WORKERS = 4


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ----------------- Text Extraction Functions -----------------

@lru_cache(maxsize=32)
def extract_text_with_pymupdf(pdf_path, page_numbers=None):
    """Extract text from PDF using PyMuPDF (fast for text-based PDFs)"""
    try:
        doc = fitz.open(pdf_path)

        if page_numbers is None:
            page_numbers = range(len(doc))

        text_by_page = {}
        for page_num in page_numbers:
            if page_num < len(doc):
                page = doc.load_page(page_num)
                blocks = page.get_text("dict")["blocks"]

                # Sort blocks by vertical position
                sorted_blocks = sorted(blocks, key=lambda b: b["bbox"][1])

                page_text = ""
                for block in sorted_blocks:
                    if "lines" in block:
                        for line in block["lines"]:
                            if "spans" in line:
                                for span in line["spans"]:
                                    if "text" in span:
                                        page_text += span["text"] + " "
                                page_text += "\n"

                text_by_page[page_num] = page_text

        doc.close()
        return text_by_page
    except Exception as e:
        logger.error(f"PyMuPDF extraction error: {str(e)}")
        return {}


def extract_text_with_pdfminer(pdf_path, page_numbers=None):
    """Extract text from PDF using pdfminer (good for complex layouts)"""
    try:
        # Set up parameters for better layout preservation
        laparams = LAParams(
            line_margin=0.5,
            word_margin=0.1,
            char_margin=2.0,
            boxes_flow=0.5,
            detect_vertical=True
        )

        if page_numbers is None:
            text = pdfminer_extract_text(pdf_path, laparams=laparams)
            return {0: text}  # Return all text as page 0 if no specific pages requested

        # For specific pages, we need a more complex approach
        from pdfminer.pdfpage import PDFPage
        from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
        from pdfminer.converter import TextConverter

        text_by_page = {}
        for page_num in page_numbers:
            output_string = io.StringIO()
            with open(pdf_path, 'rb') as in_file:
                rsrcmgr = PDFResourceManager()
                device = TextConverter(rsrcmgr, output_string, laparams=laparams)
                interpreter = PDFPageInterpreter(rsrcmgr, device)

                for i, page in enumerate(PDFPage.get_pages(in_file)):
                    if i == page_num:
                        interpreter.process_page(page)
                        break

            text_by_page[page_num] = output_string.getvalue()
            output_string.close()

        return text_by_page
    except Exception as e:
        logger.error(f"PDFMiner extraction error: {str(e)}")
        return {}


def extract_text_with_pypdf2(pdf_path, page_numbers=None):
    """Extract text from PDF using PyPDF2 (simple but reliable)"""
    try:
        text_by_page = {}
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)

            if page_numbers is None:
                page_numbers = range(len(reader.pages))

            for page_num in page_numbers:
                if page_num < len(reader.pages):
                    page = reader.pages[page_num]
                    text_by_page[page_num] = page.extract_text()

        return text_by_page
    except Exception as e:
        logger.error(f"PyPDF2 extraction error: {str(e)}")
        return {}


def extract_text_with_ocr(pdf_path, page_numbers=None):
    """Extract text from PDF using OCR (for scanned documents)"""
    try:
        images = convert_from_path(pdf_path)

        if page_numbers is None:
            page_numbers = range(len(images))

        text_by_page = {}
        for i in page_numbers:
            if i < len(images):
                # Apply preprocessing to improve OCR quality
                image = images[i]

                # Convert to grayscale if needed
                if image.mode != 'L':
                    image = image.convert('L')

                # Enhance image for better OCR (optional depending on performance)
                # from PIL import ImageEnhance
                # enhancer = ImageEnhance.Contrast(image)
                # image = enhancer.enhance(1.5)

                # Use pytesseract with improved settings
                custom_config = r'--oem 3 --psm 6'
                text = pytesseract.image_to_string(image, config=custom_config)
                text_by_page[i] = text

        return text_by_page
    except Exception as e:
        logger.error(f"OCR extraction error: {str(e)}")
        return {}


def extract_with_unstructured(pdf_path, page_numbers=None):
    """Extract structured content using unstructured library if available"""
    if not UNSTRUCTURED_AVAILABLE:
        return {}

    try:
        elements = partition_pdf(pdf_path)

        # Group elements by page
        text_by_page = {}
        for element in elements:
            page_num = getattr(element, 'page_number', 0) or 0

            if page_numbers is not None and page_num not in page_numbers:
                continue

            if page_num not in text_by_page:
                text_by_page[page_num] = ""

            element_text = str(element)
            if hasattr(element, 'element_type'):
                if element.element_type == 'Title':
                    element_text = f"<h1>{html.escape(element_text)}</h1>"
                elif element.element_type == 'NarrativeText':
                    element_text = f"<p>{html.escape(element_text)}</p>"
                elif element.element_type == 'ListItem':
                    element_text = f"<li>{html.escape(element_text)}</li>"

            text_by_page[page_num] += element_text + "\n"

        return text_by_page
    except Exception as e:
        logger.error(f"Unstructured extraction error: {str(e)}")
        return {}


def extract_tables_with_camelot(pdf_path, page_numbers=None):
    """Extract tables using camelot if available"""
    if not CAMELOT_AVAILABLE:
        return {}

    try:
        if page_numbers is None:
            # Get number of pages with PyMuPDF for faster access
            doc = fitz.open(pdf_path)
            page_numbers = list(range(1, len(doc) + 1))  # Camelot uses 1-based indexing
            doc.close()

        tables_by_page = {}
        for page_num in page_numbers:
            try:
                # Try both lattice and stream methods
                tables_lattice = camelot.read_pdf(pdf_path, pages=str(page_num), flavor='lattice')
                tables_stream = camelot.read_pdf(pdf_path, pages=str(page_num), flavor='stream')

                page_tables = []

                # Choose the best tables based on accuracy scores
                for tables in [tables_lattice, tables_stream]:
                    for table in tables:
                        if table.accuracy > 80:  # Only keep tables with good accuracy
                            html_table = table.df.to_html(index=False, header=True)
                            page_tables.append(html_table)

                # Store the tables for this page
                tables_by_page[page_num - 1] = page_tables  # Convert to 0-based indexing
            except:
                # If table extraction fails for a page, continue to the next
                continue

        return tables_by_page
    except Exception as e:
        logger.error(f"Camelot extraction error: {str(e)}")
        return {}


# ----------------- Smart PDF Analysis -----------------

def analyze_pdf_type(pdf_path):
    """Determine if a PDF is mostly text-based or scanned"""
    try:
        # Try to quickly extract text with PyMuPDF
        doc = fitz.open(pdf_path)

        # Sample a few pages for efficiency
        num_pages = len(doc)
        sample_pages = min(5, num_pages)

        # Calculate text density
        text_density = []
        for i in range(sample_pages):
            page = doc.load_page(i)
            text = page.get_text()
            text_length = len(text.strip())

            # Get page dimensions
            width, height = page.rect.width, page.rect.height
            page_area = width * height

            # Text density = characters per unit area
            if page_area > 0:
                density = text_length / page_area
                text_density.append(density)

        # Close the document
        doc.close()

        # If we have density values, use them to determine if it's scanned
        if text_density:
            avg_density = sum(text_density) / len(text_density)

            # Extremely low text density suggests a scanned document
            if avg_density < 0.01:
                return "scanned"
            # Medium-low density might be a document with images/tables
            elif avg_density < 0.05:
                return "mixed"
            # Higher density suggests mostly text
            else:
                return "text"

        return "unknown"
    except Exception as e:
        logger.error(f"PDF type analysis error: {str(e)}")
        return "unknown"


def smart_extract_text(pdf_path):
    """Extract text using the most appropriate methods based on document analysis"""
    # Analyze what kind of PDF we're dealing with
    pdf_type = analyze_pdf_type(pdf_path)
    logger.info(f"Detected PDF type: {pdf_type}")

    # Get document page count
    try:
        with fitz.open(pdf_path) as doc:
            num_pages = len(doc)
    except:
        try:
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                num_pages = len(reader.pages)
        except:
            num_pages = 1  # Fallback if we can't determine page count

    # For small documents, process all pages at once
    if num_pages <= 5:
        page_numbers = None  # Process all pages
    else:
        # For larger documents, process in batches
        page_numbers = list(range(num_pages))

    # Initialize combined text storage
    combined_text_by_page = {}
    tables_by_page = {}

    # Use different extraction methods based on PDF type
    if pdf_type == "text":
        # For text-based PDFs, use fast extractors
        extractors = [
            (extract_text_with_pymupdf, 0.7),
            (extract_text_with_pdfminer, 0.3)
        ]

        # Add optional extractors if available
        if UNSTRUCTURED_AVAILABLE:
            extractors.append((extract_with_unstructured, 0.5))

    elif pdf_type == "scanned":
        # For scanned PDFs, rely heavily on OCR
        extractors = [
            (extract_text_with_ocr, 0.9),
            (extract_text_with_pymupdf, 0.1),  # Sometimes PDFs contain hidden text
        ]

    else:  # "mixed" or "unknown"
        # Try a balanced approach
        extractors = [
            (extract_text_with_pymupdf, 0.4),
            (extract_text_with_pdfminer, 0.3),
            (extract_text_with_ocr, 0.3)
        ]

        # Add optional extractors if available
        if UNSTRUCTURED_AVAILABLE:
            extractors.append((extract_with_unstructured, 0.4))

    # Process each page with each extractor
    for extractor, weight in extractors:
        extractor_name = extractor.__name__
        logger.info(f"Running extractor: {extractor_name}")

        # Extract text
        text_results = extractor(pdf_path, page_numbers)

        # Merge results with weights
        for page_num, text in text_results.items():
            if page_num not in combined_text_by_page:
                combined_text_by_page[page_num] = {"text": "", "confidence": 0}

            # Only update if we got meaningful text
            if text and len(text.strip()) > 20:  # Minimal text length threshold
                # Weight the contribution of this extractor
                current_confidence = combined_text_by_page[page_num]["confidence"]
                new_confidence = current_confidence + weight

                # Weighted combination of texts
                if current_confidence > 0:
                    ratio = current_confidence / new_confidence
                    combined_text_by_page[page_num]["text"] = (
                            combined_text_by_page[page_num]["text"] * ratio +
                            text * (1 - ratio)
                    )
                else:
                    combined_text_by_page[page_num]["text"] = text

                combined_text_by_page[page_num]["confidence"] = new_confidence

    # Extract tables if available
    if CAMELOT_AVAILABLE:
        tables_by_page = extract_tables_with_camelot(pdf_path, page_numbers)

    # Integrate tables into text
    final_text_by_page = {}
    for page_num, page_data in combined_text_by_page.items():
        final_text = page_data["text"]

        # Add tables if available
        if page_num in tables_by_page and tables_by_page[page_num]:
            for table_html in tables_by_page[page_num]:
                # Try to find a suitable position for the table
                # For simplicity, we'll add tables at paragraph breaks
                parts = re.split(r'\n\n', final_text, 1)
                if len(parts) > 1:
                    final_text = parts[0] + "\n\n" + table_html + "\n\n" + parts[1]
                else:
                    final_text += "\n\n" + table_html

        final_text_by_page[page_num] = final_text

    # Sort by page number and combine
    sorted_pages = sorted(final_text_by_page.keys())
    combined_text = "\n\n".join(final_text_by_page[page] for page in sorted_pages)

    return combined_text


# ----------------- Text to HTML Conversion -----------------

def clean_text(text):
    """Clean extracted text for better HTML conversion"""
    if not text:
        return ""

    # Remove excessive whitespace but preserve paragraph breaks
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)

    # Fix common OCR issues
    text = re.sub(r'l\b', 'I', text)  # Lowercase l at end of word is often an uppercase I
    text = re.sub(r'\bI([^a-zA-Z\s])', 'l\\1', text)  # I followed by punctuation is often lowercase l

    return text.strip()


def detect_font_patterns(text):
    """Analyze text to find potential heading patterns based on formatting clues"""
    lines = text.split('\n')
    line_info = []

    # Extract potential formatting clues
    for line in lines:
        line = line.strip()
        if not line:
            line_info.append({
                'text': '',
                'is_uppercase': False,
                'has_number_prefix': False,
                'length': 0,
                'ends_with_colon': False
            })
            continue

        # Calculate properties
        info = {
            'text': line,
            'is_uppercase': line.isupper(),
            'has_number_prefix': bool(re.match(r'^(\d+\.)+\s', line) or re.match(r'^[IVXivx]+\.\s', line)),
            'length': len(line),
            'ends_with_colon': line.endswith(':')
        }
        line_info.append(info)

    # Identify potential headings
    headings = []
    for i, info in enumerate(line_info):
        if not info['text']:
            continue

        score = 0

        # Uppercase lines are often headings
        if info['is_uppercase']:
            score += 3

        # Numbered sections are likely headings
        if info['has_number_prefix']:
            score += 4

        # Very short or very long lines are less likely to be headings
        if 10 <= info['length'] <= 100:
            score += 1

        # Lines ending with colon might be headings or labels
        if info['ends_with_colon']:
            score += 2

        # Check if surrounded by empty lines (paragraph break)
        prev_empty = i == 0 or not line_info[i - 1]['text']
        next_empty = i == len(line_info) - 1 or not line_info[i + 1]['text']
        if prev_empty and next_empty:
            score += 2

        # Lines followed by underlines (===, ---) are headings
        if i < len(line_info) - 1 and line_info[i + 1]['text'] and re.match(r'^[=\-_]{3,}$', line_info[i + 1]['text']):
            score += 5

        # If score is high enough, mark as heading
        if score >= 4:
            level = 2 if score >= 6 else 3
            headings.append((i, level))

    return headings


def detect_lists(text):
    """Detect and convert bullet lists and numbered lists"""
    # Find bullet lists with various bullet types
    text = re.sub(r'(?m)^[•●◦○*\-]\s+(.*?)$', r'<li>\1</li>', text)

    # Find numbered lists with various formats (1., 1), a., A., i., I., etc)
    text = re.sub(r'(?m)^(\d+[\.\)]\s+)(.*?)$', r'<li>\2</li>', text)
    text = re.sub(r'(?m)^([a-z][\.\)]\s+)(.*?)$', r'<li>\2</li>', text)
    text = re.sub(r'(?m)^([A-Z][\.\)]\s+)(.*?)$', r'<li>\2</li>', text)
    text = re.sub(r'(?m)^([ivxIVX]+[\.\)]\s+)(.*?)$', r'<li>\2</li>', text)

    # Wrap list items in <ul> or <ol> tags
    soup = BeautifulSoup(f'<div>{text}</div>', 'html.parser')

    # Process list items
    list_start = None
    list_items = []
    list_type = None

    # This more complex approach handles nested and mixed lists
    for i, element in enumerate(list(soup.div.children)):
        if isinstance(element, str):
            if list_items:
                # Create the appropriate list container
                if list_type == 'ol':
                    list_tag = soup.new_tag('ol')
                else:
                    list_tag = soup.new_tag('ul')

                # Place the list at the start position
                if list_start < len(soup.div.contents):
                    soup.div.contents[list_start] = list_tag

                    # Remove list items from their original positions
                    # (offset by 1 for each item already removed)
                    for j, item in enumerate(list_items):
                        if list_start + 1 < len(soup.div.contents):
                            soup.div.contents.pop(list_start + 1)

                        # Add to the list tag
                        list_tag.append(item)

                # Reset list tracking
                list_items = []
                list_start = None
                list_type = None
        elif element.name == 'li':
            if list_start is None:
                list_start = i

            # Try to determine if it's an ordered list
            if re.match(r'^\d+\.', element.get_text().strip()):
                if list_type is None or list_type == 'ol':
                    list_type = 'ol'
                    list_items.append(element)
                else:
                    # Type conflict, finish the current list and start a new one
                    if list_items:
                        list_tag = soup.new_tag('ul')
                        soup.div.contents[list_start] = list_tag
                        for j, item in enumerate(list_items):
                            if list_start + 1 < len(soup.div.contents):
                                soup.div.contents.pop(list_start + 1)
                            list_tag.append(item)

                    list_items = [element]
                    list_start = i
                    list_type = 'ol'
            else:
                if list_type is None or list_type == 'ul':
                    list_type = 'ul'
                    list_items.append(element)
                else:
                    # Type conflict, finish the current list and start a new one
                    if list_items:
                        list_tag = soup.new_tag('ol')
                        soup.div.contents[list_start] = list_tag
                        for j, item in enumerate(list_items):
                            if list_start + 1 < len(soup.div.contents):
                                soup.div.contents.pop(list_start + 1)
                            list_tag.append(item)

                    list_items = [element]
                    list_start = i
                    list_type = 'ul'

    # Handle any remaining list
    if list_items:
        if list_type == 'ol':
            list_tag = soup.new_tag('ol')
        else:
            list_tag = soup.new_tag('ul')

        if list_start < len(soup.div.contents):
            soup.div.contents[list_start] = list_tag

            for j, item in enumerate(list_items):
                if list_start + 1 < len(soup.div.contents):
                    soup.div.contents.pop(list_start + 1)
                list_tag.append(item)

    return str(soup.div)[5:-6]  # Remove the <div> wrapper


def detect_tables_in_text(text):
    """Convert ASCII-art tables in text to HTML tables"""
    lines = text.split('\n')
    table_start = -1
    in_table = False

    # Look for consecutive lines with lots of vertical bars or plus signs
    for i, line in enumerate(lines):
        line = line.strip()

        # Check if line looks like a table separator or contains many columns
        if ('+' in line and '-' in line) or line.count('|') > 2:
            if not in_table:
                table_start = i
                in_table = True
        else:
            # If we were in a table but this line doesn't match, check if table ended
            if in_table and not (line.count('|') > 1 or not line):
                in_table = False

                # Only process if table has at least 3 lines
                if i - table_start > 2:
                    # Extract the table text
                    table_text = '\n'.join(lines[table_start:i])

                    # Convert to HTML table
                    html_table = convert_ascii_to_html_table(table_text)

                    # Replace in the original text
                    lines[table_start:i] = [html_table]

                    # Need to restart scan from the current position
                    return detect_tables_in_text('\n'.join(lines))

    return '\n'.join(lines)


def convert_ascii_to_html_table(table_text):
    """Convert ASCII-art table to HTML"""
    lines = table_text.strip().split('\n')

    # Find header separator line
    header_line = -1
    for i, line in enumerate(lines):
        if re.match(r'[-+\s]+$', line.strip()):
            header_line = i
            break

    # Split into data rows
    rows = []
    current_row = []

    for i, line in enumerate(lines):
        # Skip separator lines
        if re.match(r'[-+\s]+$', line.strip()):
            continue

        # Process data line
        cells = re.split(r'\s*\|\s*', line.strip())

        # Remove empty cells at the beginning/end from splitting edge pipes
        if cells and not cells[0].strip():
            cells = cells[1:]
        if cells and not cells[-1].strip():
            cells = cells[:-1]

        # Add to rows if we have data
        if cells:
            rows.append(cells)

    # Create HTML table
    html = ['<table class="extracted-table">']

    # Add header row if found
    if header_line > 0:
        html.append('<thead>')
        html.append('<tr>')
        for cell in rows[0]:
            html.append(f'<th>{html.escape(cell.strip())}</th>')
        html.append('</tr>')
        html.append('</thead>')
        rows = rows[1:]

    # Add data rows
    html.append('<tbody>')
    for row in rows:
        html.append('<tr>')
        for cell in row:
            html.append(f'<td>{html.escape(cell.strip())}</td>')
        html.append('</tr>')
    html.append('</tbody>')

    html.append('</table>')
    return '\n'.join(html)


def convert_text_to_html(text):
    """Convert extracted text to structured HTML with improved formatting"""
    if not text:
        return "<p>No text could be extracted from the PDF.</p>"

    # Clean the text
    text = clean_text(text)

    # Handle tables in the text (if not already processed as HTML)
    if '<table' not in text:
        text = detect_tables_in_text(text)

    # Find potential headings
    heading_positions = detect_font_patterns(text)

    # Insert heading tags
    if heading_positions:
        lines = text.split('\n')
        for position, level in reversed(heading_positions):  # Process from end to preserve positions
            if position < len(lines):
                text_content = html.escape(lines[position].strip())
                lines[position] = f'<h{level}>{text_content}</h{level}>'
        text = '\n'.join(lines)

    # Detect lists
    text = detect_lists(text)

    # Convert remaining newlines to paragraphs
    paragraphs = []
    current_paragraph = []
    html_content = []

    # Skip existing HTML elements and properly handle paragraph breaks
    # Use BeautifulSoup to help with mixed HTML and plaintext
    soup = BeautifulSoup(f'<div>{text}</div>', 'html.parser')

    for element in soup.div.contents:
        if isinstance(element, str):
            # Split text by double newlines (paragraph breaks)
            parts = re.split(r'\n\n+', str(element))

            for i, part in enumerate(parts):
                # Skip empty parts
                if not part.strip():
                    continue

                # Split part by single newlines
                lines = part.split('\n')

                # Combine lines into a single paragraph
                combined = ' '.join(line.strip() for line in lines if line.strip())

                if combined:
                    html_content.append(f'<p>{combined}</p>')

                # Add paragraph break if not the last part
                if i < len(parts) - 1:
                    html_content.append('')
        else:
            # Preserve existing HTML elements
            html_content.append(str(element))

    result = '\n'.join(html_content)

    # Properly build nested structures (unindent sublists, etc.)
    soup = BeautifulSoup(f'<div>{result}</div>', 'html.parser')

    # Final cleanup
    # Remove empty paragraphs
    for p in soup.find_all('p'):
        if not p.get_text().strip():
            p.decompose()

    return str(soup.div)[5:-6]  # Remove the <div> wrapper


# ----------------- Routes -----------------

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        # Generate unique filename
        unique_filename = str(uuid.uuid4()) + secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)

        # Determine conversion method
        conversion_method = request.form.get('conversion_method', 'auto')

        # Extract text based on the selected method
        start_time = time.time()
        text = None

        if conversion_method == 'auto':
            # Use our smart extraction pipeline
            text = smart_extract_text(file_path)
        elif conversion_method == 'text_based':
            # Extract with text-focused methods
            text = ""
            text_by_page = extract_text_with_pymupdf(file_path)
            for page_num in sorted(text_by_page.keys()):
                text += text_by_page[page_num] + "\n\n"
        elif conversion_method == 'scanned':
            # Use OCR for scanned documents
            text = ""
            text_by_page = extract_text_with_ocr(file_path)
            for page_num in sorted(text_by_page.keys()):
                text += text_by_page[page_num] + "\n\n"

        # Convert the extracted text to HTML
        html_content = convert_text_to_html(text)

        # Add processing time info
        processing_time = time.time() - start_time
        processing_info = f"<!-- Processing time: {processing_time:.2f} seconds -->"

        # Save the HTML content to a file
        html_filename = unique_filename.rsplit('.', 1)[0] + '.html'
        html_path = os.path.join(app.config['UPLOAD_FOLDER'], html_filename)

        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Converted PDF</title>
    <link rel="stylesheet" href="/static/css/style.css">
</head>
<body>
    {processing_info}
    <div class="container">
        <div class="content">
            {html_content}
        </div>
    </div>
</body>
</html>""")

        return render_template('result.html',
                               html_content=html_content,
                               download_url=url_for('download_file', filename=html_filename),
                               processing_time=f"{processing_time:.2f}")

    flash('Invalid file type. Please upload a PDF file.')
    return redirect(url_for('index'))


@app.route('/uploads/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/convert-text', methods=['POST'])
def convert_text():
    """Endpoint to convert plain text to HTML structure"""
    if 'text' not in request.form or not request.form['text'].strip():
        flash('No text provided')
        return redirect(url_for('index'))

    text = request.form['text']
    html_content = convert_text_to_html(text)

    # Generate a unique filename for the HTML
    html_filename = f"{uuid.uuid4()}.html"
    html_path = os.path.join(app.config['UPLOAD_FOLDER'], html_filename)

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Converted Text</title>
    <link rel="stylesheet" href="/static/css/style.css">
</head>
<body>
    <div class="container">
        <div class="content">
            {html_content}
        </div>
    </div>
</body>
</html>""")

    return render_template('result.html',
                           html_content=html_content,
                           download_url=url_for('download_file', filename=html_filename))


@app.route('/api/convert', methods=['POST'])
def api_convert():
    """API endpoint for conversion, returning JSON response"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        # Generate unique filename
        unique_filename = str(uuid.uuid4()) + secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)

        # Use smart extraction
        try:
            text = smart_extract_text(file_path)
            html_content = convert_text_to_html(text)

            # Generate downloadable HTML file
            html_filename = unique_filename.rsplit('.', 1)[0] + '.html'
            html_path = os.path.join(app.config['UPLOAD_FOLDER'], html_filename)

            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Converted PDF</title>
</head>
<body>
    {html_content}
</body>
</html>""")

            return jsonify({
                'success': True,
                'html': html_content,
                'download_url': url_for('download_file', filename=html_filename, _external=True)
            })

        except Exception as e:
            logger.error(f"API conversion error: {str(e)}")
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Invalid file type. Please upload a PDF file.'}), 400


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')