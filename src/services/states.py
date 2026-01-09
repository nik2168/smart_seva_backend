from typing import TypedDict, Optional

class AISummaryState(TypedDict):
    """State that flows through the LangGraph workflow"""
    file_path: str  # Path to the uploaded file
    image_bytes: Optional[bytes]  # Image bytes (set after validation/conversion)
    input_type: Optional[str]  # MIME type (detected from file extension)
    file_name: Optional[str]  # Original filename
    ocr_text: str
    ocr_confidence: float  # Average OCR confidence score (0.0-1.0)
    ocr_extracted_data: dict
    document_type: str
    errors: list[str]
