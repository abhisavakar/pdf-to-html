FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-osd \
    ghostscript \
    libmagickwand-dev \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Create a policy file for ImageMagick to read PDFs (needed for Docling)
RUN echo '<policymap> \
    <policy domain="coder" rights="read|write" pattern="PDF" /> \
    <policy domain="coder" rights="read|write" pattern="LABEL" /> \
</policymap>' > /etc/ImageMagick-6/policy.xml

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create upload directory
RUN mkdir -p uploads && chmod 777 uploads

# Expose port
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]