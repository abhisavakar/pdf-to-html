import os
import uuid
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
import fitz  # PyMuPDF
import tempfile
from pdf2image import convert_from_path
import pytesseract
from werkzeug.utils import secure_filename
import re
from bs4 import BeautifulSoup
import html

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.secret_key = 'supersecretkey'  # For flashing messages

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'pdf'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def clean_text(text):
    """Basic text cleaning to handle common PDF extraction issues"""
    # Replace multiple newlines with a single one
    text = re.sub(r'\n+', '\n', text)
    # Replace multiple spaces with a single one
    text = re.sub(r' +', ' ', text)
    return text.strip()


def detect_headers(text):
    """
    Attempt to detect headers in the text based on common patterns
    Returns text with potential headers marked with <h2> tags
    """
    # Look for numbered sections or all caps lines as potential headers
    lines = text.split('\n')
    result = []

    for line in lines:
        line = line.strip()
        if not line:
            result.append('')
            continue

        # Check for numbered sections like "1.", "1.1", "I.", "A."
        if re.match(r'^[0-9A-Z]+\.(\d+\.?)* ', line) and len(line) < 100:
            result.append(f'<h2>{html.escape(line)}</h2>')
        # Check for ALL CAPS lines that might be headers
        elif line.isupper() and len(line) < 100:
            result.append(f'<h2>{html.escape(line)}</h2>')
        else:
            result.append(html.escape(line))

    return '\n'.join(result)


def detect_lists(text):
    """Detect and convert bullet lists"""
    # Look for common bullet patterns
    text = re.sub(r'(?m)^[•●◦○*-]\s+(.*?)$', r'<li>\1</li>', text)
    # Look for numbered lists
    text = re.sub(r'(?m)^(\d+\.\s+)(.*?)$', r'<li>\2</li>', text)

    # Wrap adjacent <li> elements in <ul> tags
    soup = BeautifulSoup(f'<div>{text}</div>', 'html.parser')

    # Find consecutive li elements and wrap them in ul
    current_li = soup.find('li')
    while current_li:
        start_li = current_li
        li_elements = [start_li]

        # Collect consecutive li elements
        sibling = start_li.next_sibling
        while sibling and sibling.name == 'li':
            li_elements.append(sibling)
            sibling = sibling.next_sibling

        # If we found multiple consecutive li elements, wrap them in ul
        if len(li_elements) > 1 or (len(li_elements) == 1 and not start_li.parent.name == 'ul'):
            ul = soup.new_tag('ul')
            start_li.insert_before(ul)
            for li in li_elements:
                ul.append(li.extract())

        current_li = soup.find('li')

    return str(soup.div)[5:-6]  # Remove the <div> wrapper


def extract_text_from_pdf(pdf_path):
    """Extract text from a text-based PDF using PyMuPDF"""
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        print(f"Error extracting text with PyMuPDF: {e}")
        return None


def extract_text_from_scanned_pdf(pdf_path):
    """Extract text from a scanned PDF using OCR"""
    text = ""
    try:
        # Create temporary directory for the images
        with tempfile.TemporaryDirectory() as path:
            # Convert PDF to images
            images = convert_from_path(pdf_path)

            # Perform OCR on each image
            for i, image in enumerate(images):
                image_path = os.path.join(path, f'page_{i}.png')
                image.save(image_path, 'PNG')

                # Extract text using pytesseract
                page_text = pytesseract.image_to_string(image_path)
                text += page_text + "\n\n"

        return text
    except Exception as e:
        print(f"Error extracting text with OCR: {e}")
        return None


def convert_text_to_html(text):
    """Convert extracted text to structured HTML"""
    if not text:
        return "<p>No text could be extracted from the PDF.</p>"

    # Clean the text
    text = clean_text(text)

    # Detect potential headers
    text = detect_headers(text)

    # Detect lists
    text = detect_lists(text)

    # Convert newlines to paragraphs
    paragraphs = text.split('\n\n')
    html_content = ""

    for p in paragraphs:
        p = p.strip()
        if p:
            # Skip if it's already a header or list
            if p.startswith('<h') or p.startswith('<ul'):
                html_content += p + "\n"
            else:
                # Split by newline to check for further processing
                lines = p.split('\n')
                if len(lines) > 1:
                    # This might be a list or a paragraph with forced newlines
                    processed = detect_lists('\n'.join(lines))
                    if '<li>' in processed:
                        html_content += processed + "\n"
                    else:
                        html_content += f"<p>{' '.join(lines)}</p>\n"
                else:
                    html_content += f"<p>{p}</p>\n"

    return html_content


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
        text = None

        if conversion_method == 'text_based' or conversion_method == 'auto':
            text = extract_text_from_pdf(file_path)

        if (conversion_method == 'scanned' or
                (conversion_method == 'auto' and (text is None or len(text.strip()) < 100))):
            text = extract_text_from_scanned_pdf(file_path)

        # Convert the extracted text to HTML
        html_content = convert_text_to_html(text)

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


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')