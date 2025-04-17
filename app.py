import os
import uuid
import time
import re
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
import logging
from bs4 import BeautifulSoup

# Try to import docling - we'll handle import errors gracefully
try:
    from docling import Document
    from docling.config import LayoutConfig, ParsingConfig, TableConfig

    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    logging.error("Docling library not found. Please install with 'pip install docling'")

# Try to import camelot for additional table support
try:
    import camelot

    CAMELOT_AVAILABLE = True
except ImportError:
    CAMELOT_AVAILABLE = False

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


def extract_with_docling(pdf_path, output_format='html', optimize_tables=True, fix_spacing=True):
    """
    Extract content from PDF using Docling's AI-powered document understanding
    with enhanced table support and page spacing fixes

    Args:
        pdf_path: Path to the PDF file
        output_format: Output format ('html', 'markdown', or 'json')
        optimize_tables: Whether to optimize tables in the output
        fix_spacing: Whether to fix excessive spacing between pages

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
        # Configure Docling for better table extraction
        table_config = TableConfig(
            detection_confidence_threshold=0.3,  # Lower threshold to detect more tables
            extraction_model="tableformer"  # Use TableFormer model for better table extraction
        )

        # Configure layout analysis for better structure recognition
        layout_config = LayoutConfig(
            use_columns=True,  # Better handle multi-column layouts
            use_reading_order=True  # Respect reading order
        )

        # Configure parsing for better overall quality
        parsing_config = ParsingConfig(
            remove_headers_and_footers=True,  # Remove recurring headers/footers
            fix_rotation=True,  # Fix rotated content
            languages=["en"]  # Specify expected language
        )

        # Load the document using Docling with our custom configurations
        logger.info(f"Processing document with Docling: {pdf_path}")

        # Create a Document object from the PDF file with custom configurations
        doc = Document.from_pdf(
            pdf_path,
            table_config=table_config,
            layout_config=layout_config,
            parsing_config=parsing_config
        )

        # Process the document
        logger.info("Analyzing document structure...")

        # Get the content in the requested format
        if output_format == 'html':
            content = doc.to_html()

            # Post-process HTML content if needed
            if optimize_tables or fix_spacing:
                content = post_process_html(content, optimize_tables, fix_spacing)

        elif output_format == 'markdown':
            content = doc.to_markdown()
        elif output_format == 'json':
            content = doc.to_json()
        else:
            content = doc.to_html()  # Default to HTML
            if optimize_tables or fix_spacing:
                content = post_process_html(content, optimize_tables, fix_spacing)

        logger.info(f"Successfully extracted content in {output_format} format")
        return content

    except Exception as e:
        error_message = f"Error processing PDF with Docling: {str(e)}"
        logger.error(error_message)

        # Try to extract with Camelot for tables if available
        if CAMELOT_AVAILABLE and "table" in str(e).lower():
            try:
                logger.info("Attempting table extraction with Camelot...")
                return extract_tables_with_camelot(pdf_path)
            except Exception as camelot_error:
                logger.error(f"Camelot extraction failed: {str(camelot_error)}")

        return f"""
        <div class="error-message">
            <h2>Processing Error</h2>
            <p>{error_message}</p>
            <p>Please try a different PDF or conversion method.</p>
        </div>
        """


def post_process_html(html_content, optimize_tables=True, fix_spacing=True):
    """
    Post-process HTML content to optimize tables and fix page spacing issues

    Args:
        html_content: Original HTML content
        optimize_tables: Whether to optimize tables
        fix_spacing: Whether to fix excessive spacing between pages

    Returns:
        Processed HTML content
    """
    # Parse HTML content
    soup = BeautifulSoup(html_content, 'html.parser')

    # 1. Fix table formatting issues
    if optimize_tables:
        # Find all tables
        tables = soup.find_all('table')
        for table in tables:
            # Add a CSS class for better styling
            table['class'] = table.get('class', []) + ['extracted-table']

            # Check for empty headers and replace with reasonable content
            th_elements = table.find_all('th')
            for i, th in enumerate(th_elements):
                if not th.get_text().strip():
                    th.string = f"Column {i + 1}"

            # Check for empty rows (all cells empty)
            rows = table.find_all('tr')
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if all(not cell.get_text().strip() for cell in cells):
                    row.decompose()  # Remove empty rows

    # 2. Fix excessive spacing between pages
    if fix_spacing:
        # Look for page break indicators
        page_breaks = soup.find_all('hr', class_='pagebreak')

        if not page_breaks:  # If no specific page break class
            # Look for divs that might contain page info
            page_divs = soup.find_all('div', class_=lambda c: c and 'page' in c.lower())

            # Remove excessive margins/padding
            for div in page_divs:
                if 'style' in div.attrs:
                    div['style'] = re.sub(r'margin[^;]+(;|$)', '', div['style'])
                    div['style'] = re.sub(r'padding[^;]+(;|$)', '', div['style'])

                # Find page separators (could be hr tags or large whitespace)
                for tag in div.find_all('hr'):
                    tag.decompose()

        # Remove consecutive break elements
        br_tags = soup.find_all('br')
        for i, br in enumerate(br_tags):
            if i > 0 and br.previous_sibling == br_tags[i - 1]:
                br.decompose()

    # 3. Add any additional cleanup
    # Remove empty paragraphs
    for p in soup.find_all('p'):
        if not p.get_text().strip():
            p.decompose()

    return str(soup)


def extract_tables_with_camelot(pdf_path):
    """
    Fallback function to extract tables using Camelot
    when Docling table extraction fails
    """
    if not CAMELOT_AVAILABLE:
        return "<p>Table extraction failed. Camelot is not available for fallback extraction.</p>"

    try:
        # Extract tables using both lattice and stream methods
        tables_lattice = camelot.read_pdf(pdf_path, pages='all', flavor='lattice')
        tables_stream = camelot.read_pdf(pdf_path, pages='all', flavor='stream')

        # Combine all extracted tables
        html_parts = ["<h1>Extracted Tables</h1>"]

        # Process lattice tables (usually better for bordered tables)
        if len(tables_lattice) > 0:
            html_parts.append("<h2>Tables with Borders</h2>")
            for i, table in enumerate(tables_lattice):
                if table.df.size > 0:  # Only include non-empty tables
                    html_parts.append(f"<h3>Table {i + 1} (Page {table.page})</h3>")
                    table_html = table.df.to_html(index=False)
                    # Add CSS class
                    table_html = table_html.replace('<table', '<table class="extracted-table"')
                    html_parts.append(table_html)

        # Process stream tables (often better for non-bordered tables)
        if len(tables_stream) > 0:
            html_parts.append("<h2>Tables without Explicit Borders</h2>")
            for i, table in enumerate(tables_stream):
                # Avoid duplicates (similar to tables already extracted by lattice)
                is_duplicate = False
                for lattice_table in tables_lattice:
                    if table.page == lattice_table.page and table.df.equals(lattice_table.df):
                        is_duplicate = True
                        break

                if not is_duplicate and table.df.size > 0:
                    html_parts.append(f"<h3>Table {i + 1} (Page {table.page})</h3>")
                    table_html = table.df.to_html(index=False)
                    # Add CSS class
                    table_html = table_html.replace('<table', '<table class="extracted-table"')
                    html_parts.append(table_html)

        if len(html_parts) <= 1:
            return "<p>No tables were successfully extracted from the document.</p>"

        return "\n".join(html_parts)

    except Exception as e:
        logger.error(f"Error extracting tables with Camelot: {str(e)}")
        return f"<p>Error extracting tables: {str(e)}</p>"


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

            # Combine all pages with reduced spacing
            combined_text = "\n".join(text_parts)

            # Post-process to fix spacing and format tables
            soup = BeautifulSoup(combined_text, 'html.parser')

            # Add CSS classes to tables
            for table in soup.find_all('table'):
                table['class'] = table.get('class', []) + ['extracted-table']

            # Remove excessive breaks
            consecutive_br = False
            for br in soup.find_all('br'):
                if consecutive_br:
                    br.decompose()
                consecutive_br = True

                # Reset if next element is not a br
                if br.next_sibling and br.next_sibling.name != 'br':
                    consecutive_br = False

            return str(soup)

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
    camelot_status = "Available" if CAMELOT_AVAILABLE else "Not Installed"
    return render_template('index.html',
                           docling_status=docling_status,
                           camelot_status=camelot_status)


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

        # Get optimization options
        optimize_tables = request.form.get('optimize_tables', 'on') == 'on'
        fix_spacing = request.form.get('fix_spacing', 'on') == 'on'

        # Track processing time
        start_time = time.time()

        # Extract content using Docling
        if DOCLING_AVAILABLE:
            content = extract_with_docling(file_path, output_format, optimize_tables, fix_spacing)
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