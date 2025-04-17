document.addEventListener('DOMContentLoaded', function() {
    // Handle file input display
    const fileInput = document.getElementById('file');
    const fileInputName = document.querySelector('.file-input-name');
    const fileInputButton = document.querySelector('.file-input-button');

    if (fileInput && fileInputName && fileInputButton) {
        fileInputButton.addEventListener('click', function() {
            fileInput.click();
        });

        fileInput.addEventListener('change', function() {
            if (fileInput.files.length > 0) {
                fileInputName.textContent = fileInput.files[0].name;
            } else {
                fileInputName.textContent = 'No file chosen';
            }
        });
    }

    // Handle tabs
    const tabButtons = document.querySelectorAll('.tab-button');
    const tabContents = document.querySelectorAll('.tab-content');

    tabButtons.forEach(button => {
        button.addEventListener('click', function() {
            // Remove active class from all buttons and contents
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabContents.forEach(content => content.classList.remove('active'));

            // Add active class to clicked button
            button.classList.add('active');

            // Show corresponding content
            const tabId = button.getAttribute('data-tab');
            document.getElementById(tabId).classList.add('active');
        });
    });

    // Form validation
    const uploadForm = document.getElementById('upload-form');
    const textForm = document.getElementById('text-form');

    if (uploadForm) {
        uploadForm.addEventListener('submit', function(event) {
            const fileInput = document.getElementById('file');

            if (!fileInput.files.length) {
                event.preventDefault();
                alert('Please select a PDF file to upload.');
                return;
            }

            const file = fileInput.files[0];
            if (file.size > 16 * 1024 * 1024) { // 16MB limit
                event.preventDefault();
                alert('File size exceeds the 16MB limit.');
                return;
            }

            // Add loading state
            const submitButton = uploadForm.querySelector('button[type="submit"]');
            submitButton.disabled = true;
            submitButton.textContent = 'Converting...';
        });
    }

    if (textForm) {
        textForm.addEventListener('submit', function(event) {
            const textInput = document.getElementById('text');

            if (!textInput.value.trim()) {
                event.preventDefault();
                alert('Please enter some text to convert.');
                return;
            }

            // Add loading state
            const submitButton = textForm.querySelector('button[type="submit"]');
            submitButton.disabled = true;
            submitButton.textContent = 'Converting...';
        });
    }
});