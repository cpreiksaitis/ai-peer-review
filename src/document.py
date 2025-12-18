"""PDF document parsing and text extraction."""

import base64
from pathlib import Path

import fitz  # PyMuPDF


def load_pdf_as_base64(pdf_path: str | Path) -> str:
    """
    Load a PDF file and encode it as base64 for LLM input.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Base64-encoded PDF data URL (data:application/pdf;base64,...)
    """
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"File is not a PDF: {pdf_path}")
    
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    
    encoded = base64.b64encode(pdf_bytes).decode("utf-8")
    return f"data:application/pdf;base64,{encoded}"


def get_pdf_page_count(pdf_path: str | Path) -> int:
    """Get the number of pages in a PDF."""
    pdf_path = Path(pdf_path)
    doc = fitz.open(pdf_path)
    count = len(doc)
    doc.close()
    return count


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """
    Extract text content from a PDF file.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Extracted text content as a string

    Raises:
        FileNotFoundError: If the PDF file doesn't exist
        ValueError: If the file is not a valid PDF
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"File is not a PDF: {pdf_path}")

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise ValueError(f"Failed to open PDF: {e}") from e

    text_parts: list[str] = []

    for page_num, page in enumerate(doc, start=1):
        page_text = page.get_text()
        if page_text.strip():
            text_parts.append(f"--- Page {page_num} ---\n{page_text}")

    doc.close()

    if not text_parts:
        raise ValueError("PDF contains no extractable text")

    return "\n\n".join(text_parts)


def extract_metadata_from_pdf(pdf_path: str | Path) -> dict:
    """
    Extract metadata from a PDF file.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Dictionary containing PDF metadata (title, author, etc.)
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    metadata = doc.metadata
    doc.close()

    return {
        "title": metadata.get("title", ""),
        "author": metadata.get("author", ""),
        "subject": metadata.get("subject", ""),
        "keywords": metadata.get("keywords", ""),
        "creator": metadata.get("creator", ""),
        "producer": metadata.get("producer", ""),
        "creation_date": metadata.get("creationDate", ""),
        "modification_date": metadata.get("modDate", ""),
    }


def get_manuscript_content(pdf_path: str | Path, include_pdf_data: bool = True) -> dict:
    """
    Get full manuscript content including text, metadata, and optionally PDF data.

    Args:
        pdf_path: Path to the PDF file
        include_pdf_data: Whether to include base64-encoded PDF for vision models

    Returns:
        Dictionary with 'text', 'metadata', 'path', and optionally 'pdf_base64' keys
    """
    pdf_path = Path(pdf_path)

    result = {
        "path": str(pdf_path),
        "text": extract_text_from_pdf(pdf_path),
        "metadata": extract_metadata_from_pdf(pdf_path),
        "page_count": get_pdf_page_count(pdf_path),
    }
    
    if include_pdf_data:
        result["pdf_base64"] = load_pdf_as_base64(pdf_path)
    
    return result


def create_pdf_message_content(pdf_base64: str, text_prompt: str) -> list[dict]:
    """
    Create a message content list for LLM with PDF input.
    
    Args:
        pdf_base64: Base64-encoded PDF (data:application/pdf;base64,...)
        text_prompt: The text prompt to accompany the PDF
        
    Returns:
        List of content items for litellm message
    """
    # Ensure no data uri prefix for cleanliness, though caller might have stripped it
    if "," in pdf_base64:
        pdf_base64 = pdf_base64.split(",", 1)[-1]

    return [
        {"type": "text", "text": text_prompt},
        {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": pdf_base64,
            }
        },
    ]

