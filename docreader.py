"""Extract medical documents with Docling (preferred for reports)."""

import os
import sys


def read_document(file_path: str) -> str | None:
    """Read PDF/DOCX via Docling → markdown. Returns None on failure."""
    try:
        if not os.path.exists(file_path):
            print(f"Error: file not found: {file_path}")
            return None

        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(file_path)
        return result.document.export_to_markdown()
    except ImportError:
        print("Docling not installed. Run: pip install docling")
        return None
    except Exception as e:
        print(f"Docling extraction failed: {e}")
        return None


def read_medical_report(file_path: str, fallback_extractor=None) -> str | None:
    """
    Extract a medical report. Tries Docling first, then optional fallback callable.
    """
    content = read_document(file_path)
    if content and len(content.strip()) > 50:
        return content
    if fallback_extractor:
        return fallback_extractor(file_path)
    return content
