import dspy
import json
import logging
from src.config import settings
from src.utils.llm_config import configure_dspy

logger = logging.getLogger(__name__)

# Initialize DSPy based on configured provider
configure_dspy()

class AISummary(dspy.Signature):
    """Generate a summary of the document using AI."""
    ocr_text: str = dspy.InputField(desc="Raw text extracted from the document image using OCR")
    ai_summary: str = dspy.OutputField(
        desc="AI summary of the document in JSON format. REQUIRED fields: "
             "document_type (string), "
             "extracted_data (structured data from OCR in JSON format - IMPORTANT: Extract EVERY SINGLE field present in the OCR text, not just the reference schema fields. "
             "The reference schema {name: string, father_name: string, address: string, gender: string, dob: string, age: string, phone_number: string, date_of_issue: string} is only a guide. "
             "You must extract ALL fields found in the OCR text, including but not limited to: names, dates, addresses, numbers, IDs, codes, and any other information present. "
             "Include all extracted fields in the extracted_data object, even if they are not in the reference schema)."
    )

class AISummaryProcessor:
    def __init__(self):
        self.summarize_document = dspy.ChainOfThought(AISummary)
        logger.info("AISummary processor initialized")
    
    def summarize(self, ocr_text: str) -> dict:
        """
        Generate an AI-based summary and structured data extraction from OCR text.
        
        This function takes OCR-extracted text (`ocr_text`) and uses an AI model (through DSPy)
        to analyze the text and generate a summary with structured data extraction.
        It expects a JSON output which it cleans and parses into a Python dictionary.
        If the AI's output is not valid JSON or an error occurs, it logs and raises appropriate errors.
        
        Args:
            ocr_text: Raw text extracted from the document image using OCR
            
        Returns:
            Dictionary containing document_type and extracted_data
            
        Raises:
            ValueError: If the AI output is not valid JSON
            Exception: For any other errors during processing
        """
        json_str = None
        try:
            # Single DSPy call for summarization
            result = self.summarize_document(ocr_text=ocr_text)
            
            # Parse the JSON result
            json_str = result.ai_summary
            
            # Clean JSON string (remove markdown code blocks if present)
            if json_str.startswith("```json"):
                json_str = json_str.replace("```json", "").replace("```", "").strip()
            elif json_str.startswith("```"):
                json_str = json_str.replace("```", "").strip()
            
            summary_data = json.loads(json_str)
            
            return summary_data
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse summary result: {e}")
            if json_str:
                logger.error(f"Raw output: {json_str}")
            raise ValueError(f"Invalid JSON output from summarization: {e}")
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            raise

ai_summary_processor = AISummaryProcessor()