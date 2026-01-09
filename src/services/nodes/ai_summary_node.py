import logging
from src.services.states import AISummaryState
from src.services.processors.ai_summary_processor import ai_summary_processor

logger = logging.getLogger(__name__)

def summarize_document(state: AISummaryState) -> AISummaryState:
    """Generate AI summary and extract structured data from OCR text"""
    logger.info("Running AI document summarization")
    try:
        # Check if OCR text is available
        if not state.get("ocr_text"):
            raise ValueError("OCR text not available")
        
        # Run AI summarization
        summary_result = ai_summary_processor.summarize(
            ocr_text=state["ocr_text"]
        )
        
        # Extract and store results in state
        state["document_type"] = summary_result.get("document_type", "unknown")
        state["ocr_extracted_data"] = summary_result.get("extracted_data", {})
        
        logger.info(f"AI summarization completed: {state['document_type']}")
    except Exception as e:
        logger.error(f"AI document summarization failed: {e}")
        state["errors"] = state.get("errors", []) + [f"AI summarization failed: {str(e)}"]
        # Set defaults on error
        state["document_type"] = "unknown"
        state["ocr_extracted_data"] = {}
    return state

