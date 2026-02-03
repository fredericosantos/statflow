"""
Archive generation logic for data export functionality.

This module handles creating ZIP archives with exported files.

archive_generator.py
├── generate_zip_archive()  # Creates ZIP archive with all export files.
└── Archive creation and compression logic
"""

def generate_zip_archive(files: dict[str, str], zip_name: str) -> bytes:
    """Generate ZIP archive from files.

    Args:
        files: Dict of file contents.
        zip_name: Name of ZIP file.

    Returns:
        ZIP file bytes.
    """
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
        for name, content in files.items():
            zip_file.writestr(name, content)
    zip_buffer.seek(0)
    return zip_buffer.getvalue()