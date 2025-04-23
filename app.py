import os
import uuid
import time
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
import logging

# Try to import docling - we'll handle import errors gracefully
try:
    from docling import Document, load_document_model

    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    logging.error("Docling library not found. Please install with 'pip install docling'")

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max upload
app.secret_key = os.environ.get('SECRET_KEY', 'supersecretkey')

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'pdf'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_with_docling(pdf_path, conversion_method='auto'):
    """
    Extract content from PDF using Docling's AI-powered document understanding

    Args:
        pdf_path: Path to the PDF file
        conversion_method: 'auto', 'text_based', or 'scanned'

    Returns:
        Extracted HTML content
    """
    if not DOCLING_AVAILABLE:
        return f"""
        <div class="error-message">
            <h2>Docling Not Available</h2>
            <p>The Docling library is not installed. Please install it with:</p>
            <pre>pip install docling</pre>
            <p>For more information, visit: <a href="https://github.com/docling-project/docling">https://github.com/docling-project/docling</a></p>
        </div>
        """

    try:
        # Load the document using Docling
        logger.info(f"Processing document with Docling: {pdf_path}")

        # For scanned documents, use OCR
        use_ocr = conversion_method == 'scanned'

        # Preload the document model to avoid repeat loading
        # This is only needed once per application start
        _ = load_document_model()

        # Create a Document object from the PDF file with appropriate settings
        doc = Document.from_pdf(
            pdf_path,
            use_ocr=use_ocr,
            extract_images=True,
            detect_reading_order=True,
            detect_tables=True
        )

        # Process the document
        logger.info("Analyzing document structure...")

        # Generate HTML content
        html_content = doc.to_html(
            include_images=True,  # Include images in the HTML output
            include_tables=True,  # Include tables in the HTML output
            style="document"  # Use document-style formatting
        )

        logger.info("Successfully extracted content in HTML format")
        return html_content

    except Exception as e:
        error_message = f"Error processing PDF with Docling: {str(e)}"
        logger.error(error_message)
        return f"""
        <div class="error-message">
            <h2>Processing Error</h2>
            <p>{error_message}</p>
            <p>Please try a different PDF or conversion method.</p>
        </div>
        """


def fallback_extraction(pdf_path):
    """
    A simple fallback extraction method if Docling isn't available or fails
    Uses PyMuPDF if available, otherwise returns an error message
    """
    try:
        import fitz  # PyMuPDF

        text_parts = []
        try:
            # Open the PDF
            doc = fitz.open(pdf_path)

            # Process each page
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text("html")
                text_parts.append(text)

            doc.close()

            # Combine all pages
            combined_html = "\n".join(text_parts)

            # Wrap in basic HTML structure if not already wrapped
            if not combined_html.strip().startswith("<!DOCTYPE html>"):
                combined_html = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <meta charset="UTF-8">
                    <title>Converted PDF</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
                        img {{ max-width: 100%; }}
                        table {{ border-collapse: collapse; width: 100%; }}
                        table, th, td {{ border: 1px solid #ddd; }}
                        th, td {{ padding: 8px; text-align: left; }}
                    </style>
                </head>
                <body>
                    {combined_html}
                </body>
                </html>
                """

            return combined_html

        except Exception as e:
            return f"""
            <div class="error-message">
                <h2>Fallback Extraction Failed</h2>
                <p>Error: {str(e)}</p>
            </div>
            """
    except ImportError:
        return """
        <div class="error-message">
            <h2>Extraction Failed</h2>
            <p>Docling is not available, and no fallback method could be used.</p>
            <p>Please install PyMuPDF (fitz) or Docling:</p>
            <pre>pip install PyMuPDF</pre>
            <p>or</p>
            <pre>pip install docling</pre>
        </div>
        """


@app.route('/')
def index():
    docling_status = "Available" if DOCLING_AVAILABLE else "Not Installed"
    return render_template('index.html', docling_status=docling_status)


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))

    file = request.files['file']

    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))

    if file and allowed_file(file.filename):
        # Generate unique filename
        unique_filename = str(uuid.uuid4()) + secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)

        # Get conversion method from form
        conversion_method = request.form.get('conversion_method', 'auto')

        # Track processing time
        start_time = time.time()

        # Extract content using Docling
        if DOCLING_AVAILABLE:
            content = extract_with_docling(file_path, conversion_method)
        else:
            # Use fallback if Docling isn't available
            content = fallback_extraction(file_path)

        # Calculate processing time
        processing_time = time.time() - start_time

        # Save the content to an HTML file
        output_filename = unique_filename.rsplit('.', 1)[0] + '.html'
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)

        # Return the result page
        return render_template('result.html',
                               html_content=content,
                               download_url=url_for('download_file', filename=output_filename),
                               processing_time=f"{processing_time:.2f}")

    flash('Invalid file type. Please upload a PDF file.')
    return redirect(url_for('index'))


@app.route('/convert-text', methods=['POST'])
def convert_text():
    """Endpoint to convert plain text to HTML structure"""
    text = request.form.get('text', '')

    if not text.strip():
        flash('Please enter some text to convert')
        return redirect(url_for('index'))

    # Simple conversion of plain text to HTML
    html_content = text.replace('\n', '<br>').replace('  ', '&nbsp;&nbsp;')
    html_content = f'<div class="converted-text">{html_content}</div>'

    # Generate a unique filename
    unique_filename = f"{str(uuid.uuid4())}.html"
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

    # Save the HTML content
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    return render_template('result.html',
                           html_content=html_content,
                           download_url=url_for('download_file', filename=unique_filename))


@app.route('/uploads/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)


if __name__ == '__main__':
    print(f"Docling available: {DOCLING_AVAILABLE}")
    app.run(debug=True, host='0.0.0.0')