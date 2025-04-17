import os
import uuid
import time
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
import logging

# Try to import docling - we'll handle import errors gracefully
try:
    from docling import Document

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


def extract_with_docling(pdf_path, output_format='html'):
    """
    Extract content from PDF using Docling's AI-powered document understanding

    Args:
        pdf_path: Path to the PDF file
        output_format: Output format ('html', 'markdown', or 'json')

    Returns:
        Extracted content in the specified format
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

        # Create a Document object from the PDF file
        doc = Document.from_pdf(pdf_path)

        # Process the document
        logger.info("Analyzing document structure...")

        # Get the content in the requested format
        if output_format == 'html':
            content = doc.to_html()
        elif output_format == 'markdown':
            content = doc.to_markdown()
        elif output_format == 'json':
            content = doc.to_json()
        else:
            content = doc.to_html()  # Default to HTML

        logger.info(f"Successfully extracted content in {output_format} format")
        return content

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
            combined_text = "\n".join(text_parts)
            return combined_text

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

        # Determine output format
        output_format = request.form.get('output_format', 'html')

        # Track processing time
        start_time = time.time()

        # Extract content using Docling
        if DOCLING_AVAILABLE:
            content = extract_with_docling(file_path, output_format)
        else:
            # Use fallback if Docling isn't available
            content = fallback_extraction(file_path)
            output_format = 'html'  # Force HTML for fallback

        # Calculate processing time
        processing_time = time.time() - start_time

        # Save the content to a file with appropriate extension
        if output_format == 'html':
            output_filename = unique_filename.rsplit('.', 1)[0] + '.html'
            mime_type = 'text/html'
        elif output_format == 'markdown':
            output_filename = unique_filename.rsplit('.', 1)[0] + '.md'
            mime_type = 'text/markdown'
        elif output_format == 'json':
            output_filename = unique_filename.rsplit('.', 1)[0] + '.json'
            mime_type = 'application/json'
        else:
            output_filename = unique_filename.rsplit('.', 1)[0] + '.txt'
            mime_type = 'text/plain'

        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)

        # For HTML content, wrap it in a proper HTML document for preview
        if output_format == 'html':
            preview_content = content
        elif output_format == 'markdown':
            # For markdown, show the raw markdown
            preview_content = f"<pre class='markdown-preview'>{content}</pre>"
        elif output_format == 'json':
            # For JSON, format it nicely in a pre tag
            preview_content = f"<pre class='json-preview'>{content}</pre>"
        else:
            preview_content = f"<pre>{content}</pre>"

        return render_template('result.html',
                               html_content=preview_content,
                               raw_content=content,
                               download_url=url_for('download_file', filename=output_filename),
                               output_format=output_format,
                               processing_time=f"{processing_time:.2f}")

    flash('Invalid file type. Please upload a PDF file.')
    return redirect(url_for('index'))


@app.route('/convert-text', methods=['POST'])
def convert_text():
    """Endpoint to convert plain text to HTML structure (not supported with Docling)"""
    flash('Text to HTML conversion is not supported with the Docling implementation. Please upload a PDF file.')
    return redirect(url_for('index'))


@app.route('/uploads/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')