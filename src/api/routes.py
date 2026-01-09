from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
import logging
import tempfile
import os
from pathlib import Path
from src.api.models import AISummaryResponse, BatchSummaryResponse
from src.services.workflow import ai_summary_graph
from src.services.states import AISummaryState
from src.config import logger
from src.utils.ai_validation_summary import ai_validation_summary_processor

router = APIRouter(prefix="/api", tags=["ai_summary"])


@router.post("/summarize", response_model=BatchSummaryResponse)
async def summarize_documents(
    files: List[UploadFile] = File(..., description="Document files (images or PDFs)")
):
    """
    Summarize multiple documents: Extract OCR text and generate AI summary with structured data
    Supports: Images (JPEG, PNG, etc.) and PDF files
    Processes files sequentially and returns aggregated results
    """
    if not files or len(files) == 0:
        raise HTTPException(status_code=400, detail="At least one file is required")
    
    # Supported file types
    supported_types = [
        "image/jpeg", "image/jpg", "image/png", "image/gif", 
        "image/bmp", "image/webp", "application/pdf"
    ]
    
    results = []
    
    # Process each file sequentially
    for file_index, file in enumerate(files):
        try:
            # Validate file type
            if not file.content_type:
                results.append({
                    "file_name": file.filename or f"file_{file_index}",
                    "success": False,
                    "error": "File content type not specified",
                    "document_type": None,
                    "extracted_data": None,
                    "ocr_text": None,
                    "errors": ["File content type not specified"]
                })
                continue
            
            if not any(file.content_type.lower().startswith(t) for t in supported_types):
                results.append({
                    "file_name": file.filename or f"file_{file_index}",
                    "success": False,
                    "error": f"Unsupported file type: {file.content_type}",
                    "document_type": None,
                    "extracted_data": None,
                    "ocr_text": None,
                    "errors": [f"Unsupported file type: {file.content_type}"]
                })
                continue
            
            # Read file bytes and save to temporary file
            file_bytes = await file.read()
            logger.info(f"Processing document {file_index + 1}/{len(files)}: {file.filename} (type: {file.content_type})")
            
            # Determine file extension from content type or filename
            file_ext = Path(file.filename or "temp").suffix.lower() if file.filename else ""
            if not file_ext:
                # Try to infer from content type
                content_type_map = {
                    "image/jpeg": ".jpg",
                    "image/jpg": ".jpg",
                    "image/png": ".png",
                    "image/gif": ".gif",
                    "image/bmp": ".bmp",
                    "image/webp": ".webp",
                    "application/pdf": ".pdf"
                }
                file_ext = content_type_map.get(file.content_type.lower(), ".tmp")
            
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                tmp_file.write(file_bytes)
                tmp_file_path = tmp_file.name
            
            try:
                # Initialize state for LangGraph workflow
                initial_state: AISummaryState = {
                    "file_path": tmp_file_path,
                    "image_bytes": None,  # Will be set by validate_input node
                    "input_type": None,  # Will be detected by validate_input node
                    "file_name": file.filename,
                    "ocr_text": "",
                    "ocr_confidence": 0.0,
                    "ocr_extracted_data": {},
                    "document_type": "",
                    "errors": []
                }
                
                # Run the AI summary workflow
                final_state = ai_summary_graph.invoke(initial_state)
                
                # Get summary results from state
                errors = final_state.get("errors", [])
                document_type = final_state.get("document_type", "unknown")
                extracted_data = final_state.get("ocr_extracted_data", {})
                ocr_text = final_state.get("ocr_text", "")
                ocr_confidence = final_state.get("ocr_confidence", 0.0)
                
                logger.info(f"AI summary completed for {file.filename} - Document: {document_type}, OCR Confidence: {ocr_confidence:.2f}")
                
                results.append({
                    "file_name": file.filename or f"file_{file_index}",
                    "success": True,
                    "error": None,
                    "document_type": document_type,
                    "extracted_data": extracted_data,
                    "ocr_text": ocr_text if ocr_text else None,
                    "ocr_confidence": float(ocr_confidence) if ocr_confidence else None,
                    "errors": errors if errors else None
                })
            finally:
                # Clean up temporary file
                try:
                    if os.path.exists(tmp_file_path):
                        os.unlink(tmp_file_path)
                except Exception as e:
                    logger.warning(f"Failed to delete temporary file {tmp_file_path}: {e}")
        
        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {e}", exc_info=True)
            results.append({
                "file_name": file.filename or f"file_{file_index}",
                "success": False,
                "error": str(e),
                "document_type": None,
                "extracted_data": None,
                "ocr_text": None,
                "errors": [str(e)]
            })
    
    # Generate AI validation summary for all successful documents
    validation_results = {}
    try:
        logger.info("Generating AI validation summary across all documents")
        validation_results = ai_validation_summary_processor.validate(results)
        total_issues = sum(len(issues) for issues in validation_results.values()) if validation_results else 0
        logger.info(f"AI validation completed: {total_issues} mismatches found across {len(validation_results)} document types")
    except Exception as e:
        logger.error(f"AI validation failed: {e}", exc_info=True)
        # Continue even if validation fails - don't break the response
    
    return BatchSummaryResponse(results=results, validation_results=validation_results if validation_results else None)

@router.get("/health")
async def health():
    return {"status": "healthy"}