from pydantic import BaseModel
from typing import Dict, Any, Optional, List


class AISummaryResponse(BaseModel):
    document_type: str
    extracted_data: Dict[str, Any]  # Structured data extracted from OCR
    ocr_text: Optional[str] = None  # Raw OCR extracted text
    errors: Optional[list[str]] = None


class DocumentResult(BaseModel):
    file_name: str
    success: bool
    error: Optional[str] = None
    document_type: Optional[str] = None
    extracted_data: Optional[Dict[str, Any]] = None
    ocr_text: Optional[str] = None
    ocr_confidence: Optional[float] = None  # Average OCR confidence score (0.0-1.0)
    errors: Optional[list[str]] = None


class BatchSummaryResponse(BaseModel):
    results: List[DocumentResult]
    validation_results: Optional[Dict[str, List[str]]] = None  # AI validation mismatch results grouped by document type