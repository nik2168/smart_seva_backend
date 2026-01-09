import dspy
import json
import logging
from src.config import settings
from src.utils.llm_config import configure_dspy

logger = logging.getLogger(__name__)

# Initialize DSPy based on configured provider
configure_dspy()

class DocumentVerification(dspy.Signature):
    """Comprehensive document verification: identify type, check originality, and verify form fields."""
    ocr_text: str = dspy.InputField(desc="Raw text extracted from the document image using OCR")
    form_fields: str = dspy.InputField(desc="Form fields to verify in JSON format")
    verification_result: str = dspy.OutputField(
        desc="Complete verification result in JSON format. REQUIRED fields: "
             "document_type (string), "
             "document_type_confidence (float between 0.0 and 1.0, REQUIRED - confidence score for document type detection), "
             "is_original (boolean), "
             "originality_score (float between 0.0 and 1.0, REQUIRED - confidence score for how original the document seems), "
             "originality_confidence (string: 'high', 'medium', 'low'), "
             "originality_reasons (array of strings, don't use visual matches just try to match the text formatting and content), "
             "form_verification (object with field-by-field verification: "
             "{'field_name': {'match': boolean, 'ocr_value': string, 'form_value': string, 'confidence': string}}), "
             "extracted_data (structured data from OCR in JSON format). "
             "IMPORTANT: Both originality_score and document_type_confidence must be float values between 0.0 and 1.0. "
    )

class VerificationProcessor:
    def __init__(self):
        self.verify_document = dspy.ChainOfThought(DocumentVerification)
        logger.info("Verification processor initialized")
    
    def _calculate_verification_status(self, form_verification: dict, is_original: bool) -> str:
        """
        Calculate verification status based on match percentage and originality.
        
        Status rules:
        - "verified" if 95%+ matched and document is original
        - "manual_review" if 60-94% matched and document is original
        - "rejected" if <60% matched or document is not original
        
        Args:
            form_verification: Dictionary of form field verification results
            is_original: Boolean indicating if document is original
            
        Returns:
            Verification status: "verified", "manual_review", or "rejected"
        """
        # If document is not original, reject
        if not is_original:
            return "rejected"
        
        # If no form fields to verify, reject
        if not form_verification:
            return "rejected"
        
        # Calculate match percentage
        total_fields = 0
        matched_fields = 0
        
        for field_data in form_verification.values():
            if isinstance(field_data, dict):
                total_fields += 1
                if field_data.get("match", False):
                    matched_fields += 1
        
        if total_fields == 0:
            return "rejected"
        
        match_percentage = (matched_fields / total_fields) * 100
        
        # Determine status based on match percentage
        if match_percentage >= 95:
            return "verified"
        elif match_percentage >= 60:
            return "manual_review"
        else:
            return "rejected"
    
    def verify(self, ocr_text: str, form_fields: dict) -> dict:
        """
        Single comprehensive verification call that:
        1. Identifies document type
        2. Checks if document is original
        3. Verifies form fields against OCR data
        4. Returns structured extracted data
        
        Args:
            ocr_text: Raw text extracted from document image
            form_fields: Dictionary of form fields to verify
            
        Returns:
            Dictionary with all verification results
        """
        try:
            # Convert form_fields to JSON string
            form_fields_json = json.dumps(form_fields, indent=2)
            
            # Single DSPy call for all verification tasks
            result = self.verify_document(ocr_text=ocr_text, form_fields=form_fields_json)
            
            # Parse the JSON result
            json_str = result.verification_result
            
            # Clean JSON string (remove markdown code blocks if present)
            if json_str.startswith("```json"):
                json_str = json_str.replace("```json", "").replace("```", "").strip()
            elif json_str.startswith("```"):
                json_str = json_str.replace("```", "").strip()
            
            verification_data = json.loads(json_str)
            
            # Log if confidence scores are missing
            if "originality_score" not in verification_data:
                logger.warning("originality_score not found in verification result, defaulting to 0.0")
                verification_data["originality_score"] = 0.0
            if "document_type_confidence" not in verification_data:
                logger.warning("document_type_confidence not found in verification result, defaulting to 0.0")
                verification_data["document_type_confidence"] = 0.0
            
            # Calculate verification status based on match percentage
            verification_data["verification_status"] = self._calculate_verification_status(
                verification_data.get("form_verification", {}),
                verification_data.get("is_original", False)
            )
            
            logger.info(f"Document verification completed successfully - originality_score: {verification_data.get('originality_score', 0.0):.2f}, document_type_confidence: {verification_data.get('document_type_confidence', 0.0):.2f}, status: {verification_data.get('verification_status', 'rejected')}")
            return verification_data
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse verification result: {e}")
            logger.error(f"Raw output: {json_str}")
            raise ValueError(f"Invalid JSON output from verification: {e}")
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            raise

verification_processor = VerificationProcessor()