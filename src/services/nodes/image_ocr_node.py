import logging
from src.services.states import AISummaryState
from src.services.processors.image_processor import ocr_service

logger = logging.getLogger(__name__)

def extract_ocr(state: AISummaryState) -> AISummaryState:
    """Step 1: Extract text from image using OCR"""
    logger.info("Step 1: Extracting OCR text from image")

        # Check if image_bytes is available
    if not state.get("image_bytes"):
        error_msg = "Image bytes not available - input validation may have failed"
        logger.error(error_msg)
        state["errors"] = state.get("errors", []) + [error_msg]
        state["ocr_text"] = ""
        state["ocr_confidence"] = 0.0
        return state

    try:
        # Extract text and confidence from image bytes
        ocr_text, ocr_confidence = ocr_service.extract_text(state["image_bytes"])
        state["ocr_text"] = ocr_text
        state["ocr_confidence"] = ocr_confidence
        logger.info(f"OCR extraction completed: {len(ocr_text)} characters, confidence: {ocr_confidence:.2f}")
        
        # Check if OCR text is empty
        if not ocr_text or len(ocr_text.strip()) == 0:
            logger.warning("OCR extraction returned empty text")
            state["errors"] = state.get("errors", []) + ["OCR extraction returned empty text - no text detected in image"]
    except Exception as e:
        logger.error(f"OCR extraction failed: {e}", exc_info=True)
        state["errors"] = state.get("errors", []) + [f"OCR extraction failed: {str(e)}"]
        state["ocr_text"] = ""  # Set empty string on error
        state["ocr_confidence"] = 0.0  # Set confidence to 0 on error
    return state