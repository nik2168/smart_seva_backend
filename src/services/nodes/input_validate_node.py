from pdf2image import convert_from_path
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)
import logging
from PIL import Image
import io
from pathlib import Path
from src.services.states import AISummaryState

logger = logging.getLogger(__name__)

# Supported file types
SUPPORTED_IMAGE_EXTENSIONS = ['.jpeg', '.jpg', '.png', '.gif', '.bmp', '.webp']
SUPPORTED_PDF_EXTENSION = '.pdf'

# MIME type mapping
EXTENSION_TO_MIME = {
    '.jpeg': 'image/jpeg',
    '.jpg': 'image/jpeg',
    '.png': 'image/png',
    '.gif': 'image/gif',
    '.bmp': 'image/bmp',
    '.webp': 'image/webp',
    '.pdf': 'application/pdf'
}

def detect_file_type(file_path: str) -> tuple[str, str]:
    """
    Detect file type from file extension.
    
    Returns:
        tuple: (file_extension, mime_type)
    """
    path = Path(file_path)
    ext = path.suffix.lower()
    mime_type = EXTENSION_TO_MIME.get(ext, 'application/octet-stream')
    return ext, mime_type

def validate_input(state: AISummaryState) -> AISummaryState:
    """
    Validate input file type from file path and convert PDFs to images if needed.
    Reads file from path, detects type, and ensures image_bytes is always available for OCR processing.
    """
    try:
        file_path = state.get("file_path")
        
        if not file_path:
            state["errors"] = state.get("errors", []) + ["No file path provided"]
            return state
        
        # Check if file exists
        path = Path(file_path)
        if not path.exists():
            state["errors"] = state.get("errors", []) + [f"File not found: {file_path}"]
            return state
        
        # Detect file type from extension
        file_ext, mime_type = detect_file_type(file_path)
        state["input_type"] = mime_type
        
        logger.info(f"Processing file: {file_path} (type: {mime_type})")
        
        # Handle image files
        if file_ext in SUPPORTED_IMAGE_EXTENSIONS:
            logger.info(f"Input is an image: {mime_type}")
            try:
                # Read image file
                with open(file_path, "rb") as f:
                    image_bytes = f.read()
                
                # Verify it's a valid image
                Image.open(io.BytesIO(image_bytes))
                state["image_bytes"] = image_bytes
                logger.info("Image validation successful")
            except Exception as e:
                state["errors"] = state.get("errors", []) + [f"Invalid image file: {str(e)}"]
            return state
        
        # Handle PDF files
        elif file_ext == SUPPORTED_PDF_EXTENSION:
            logger.info("Input is a PDF, converting to image...")
            try:
                # Convert PDF file to images
                converted_images = convert_from_path(file_path)
                
                if len(converted_images) == 0:
                    state["errors"] = state.get("errors", []) + ["No pages found in PDF"]
                    return state
                
                # Use first page for now (you might want to handle multiple pages)
                first_page = converted_images[0]
                
                # Convert PIL Image to bytes
                img_byte_arr = io.BytesIO()
                first_page.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()
                
                # Update state with converted image
                state["image_bytes"] = img_byte_arr
                state["input_type"] = "image/png"  # Update to reflect conversion
                logger.info(f"PDF converted to image successfully ({len(converted_images)} pages, using first page)")
                
            except PDFInfoNotInstalledError:
                state["errors"] = state.get("errors", []) + [
                    "PDF processing requires poppler-utils. Install with: apt-get install poppler-utils (Linux) or brew install poppler (Mac)"
                ]
            except (PDFPageCountError, PDFSyntaxError) as e:
                state["errors"] = state.get("errors", []) + [f"Invalid PDF file: {str(e)}"]
            except Exception as e:
                logger.error(f"Error converting PDF to image: {e}", exc_info=True)
                state["errors"] = state.get("errors", []) + [f"Error converting PDF: {str(e)}"]
            
            return state
        
        # Unsupported file type
        else:
            state["errors"] = state.get("errors", []) + [
                f"Unsupported file type: {file_ext}. Supported types: {', '.join(SUPPORTED_IMAGE_EXTENSIONS + [SUPPORTED_PDF_EXTENSION])}"
            ]
            return state
            
    except Exception as e:
        logger.error(f"Error validating input: {e}", exc_info=True)
        state["errors"] = state.get("errors", []) + [f"Error validating input: {str(e)}"]
        return state