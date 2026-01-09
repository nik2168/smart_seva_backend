import dspy
import json
import logging
from typing import List, Dict, Any
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

class AIValidation(dspy.Signature):
    """Validate and compare data across multiple documents to find mismatches."""
    documents_data: str = dspy.InputField(
        desc="JSON string containing array of documents with their extracted_data, document_type, and file_name. "
             "Each document has: file_name, document_type, extracted_data (object with all extracted fields)"
    )
    validation_results: str = dspy.OutputField(
        desc="Return a JSON object where keys are document types (lowercase, e.g., 'adhaar', 'pan', 'birth_certificate', 'driving_license', 'application') "
             "and values are arrays of natural, conversational validation issue descriptions. "
             ""
             "CRITICAL VALIDATION RULES: "
             "1. Only compare fields that are semantically the same (e.g., 'date_of_birth' with 'dob', 'name' with 'full_name', but NOT 'date_of_birth' with 'id_number' or 'registration_number') "
             "2. Before comparing, validate that values match expected formats: "
             "   - Dates should look like dates (DD-MM-YYYY, DD/MM/YYYY, YYYY-MM-DD, etc.) - NOT random numbers like '07/0412007' "
             "   - ID numbers should be numeric/alphanumeric strings, not dates "
             "   - Names should contain letters, not just numbers "
             "3. If a value doesn't match the expected format for its field type, DO NOT compare it - skip that comparison "
             "4. Only report mismatches when comparing valid, correctly formatted values of the same field type "
             ""
             "Write each issue description as a clear, natural sentence that reads like a human explaining the problem. "
             "Each description should: "
             "- Identify the specific field with the issue (name, date of birth, address, ID number, etc.) "
             "- State what value is present (or note if it's missing/empty) "
             "- Compare it with other documents when relevant (ONLY if both values are valid and same field type) "
             "- Explain why it's a concern in plain language "
             ""
             "Examples of natural issue descriptions: "
             "- 'Name mismatch: shows \"Rohan Kumar\" but PAN card has \"Rohan K. Sharma\" - middle name differs' "
             "- 'Date of birth doesn't match: shows 20-06-1986 but PAN card shows 20-06-1990' "
             "- 'Address is different from Aadhaar: shows \"123 Main St, Delhi\" while Aadhaar has \"456 Park Ave, Mumbai\"' "
             "- 'Aadhaar number field is empty - required for verification' "
             "- 'Name field is missing - cannot verify identity without name' "
             "- 'Phone number format differs: shows 9876543210 but application has +91-98765-43210 (same number, different format)' "
             ""
             "DO NOT report issues for: "
             "- Comparing date fields with non-date values (e.g., '07/0412007' is NOT a date - don't compare it with actual dates) "
             "- Comparing ID numbers with dates "
             "- Comparing fields that are clearly different types even if field names are similar "
             ""
             "Output format: "
             "{"
             "  'adhaar': ['Natural issue description 1', 'Natural issue description 2'], "
             "  'pan': ['Natural issue description'], "
             "  'application': ['Missing Aadhaar number', 'Name field is empty']"
             "} "
             ""
             "Important: Write in natural, conversational language. Avoid repetitive phrases like 'it has' or 'while the'. "
             "Each issue should be a standalone, clear sentence. Only report real mismatches between valid, comparable fields. "
             "Return empty object {} if no mismatches found."
    )

class AIValidationSummary:
    def __init__(self):
        self.summarize_document = dspy.ChainOfThought(AISummary)
        self.validate_documents = dspy.ChainOfThought(AIValidation)
        logger.info("AIValidationSummary processor initialized")
    
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
    
    def validate(self, documents: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """
        Validate and compare data across multiple documents to find mismatches.
        
        This function takes a list of document results (each with extracted_data, document_type, file_name)
        and uses AI to compare fields across documents, identifying mismatches in critical fields like
        name, date of birth, address, etc.
        
        Args:
            documents: List of dictionaries, each containing:
                - file_name: str
                - document_type: str
                - extracted_data: dict (with extracted fields)
                - success: bool (only successful documents should be included)
        
        Returns:
            Dictionary where keys are document types (lowercase) and values are lists of validation mismatch strings
            Example: {"adhaar": ["issue 1", "issue 2"], "pan": ["issue 1"]}
            
        Raises:
            ValueError: If the AI output is not valid JSON
            Exception: For any other errors during processing
        """
        # Filter only successful documents with extracted data
        valid_documents = [
            {
                "file_name": doc.get("file_name", "unknown"),
                "document_type": doc.get("document_type", "unknown"),
                "extracted_data": doc.get("extracted_data", {})
            }
            for doc in documents
            if doc.get("success", False) and doc.get("extracted_data")
        ]
        
        # If less than 2 documents, no comparison possible
        if len(valid_documents) < 2:
            logger.info("Less than 2 valid documents for validation, returning empty results")
            return {}
        
        json_str = None
        try:
            # Prepare documents data as JSON string
            documents_json = json.dumps(valid_documents, indent=2)
            
            logger.info(f"Validating {len(valid_documents)} documents for mismatches")
            
            # Call AI validation
            result = self.validate_documents(documents_data=documents_json)
            
            # Parse the JSON result
            json_str = result.validation_results
            
            # Clean JSON string (remove markdown code blocks if present)
            if json_str.startswith("```json"):
                json_str = json_str.replace("```json", "").replace("```", "").strip()
            elif json_str.startswith("```"):
                json_str = json_str.replace("```", "").strip()
            
            # Parse validation results
            validation_results = json.loads(json_str)
            
            # Ensure it's a dictionary
            if isinstance(validation_results, dict):
                # Normalize document type keys to lowercase
                normalized_results = {}
                total_issues = 0
                for doc_type, issues in validation_results.items():
                    normalized_key = doc_type.lower().replace(" ", "_")
                    if isinstance(issues, list):
                        normalized_results[normalized_key] = issues
                        total_issues += len(issues)
                    else:
                        # If value is not a list, convert it to a list
                        normalized_results[normalized_key] = [str(issues)]
                        total_issues += 1
                
                logger.info(f"Found {total_issues} validation mismatches across {len(normalized_results)} document types")
                return normalized_results
            elif isinstance(validation_results, list):
                # Fallback: if AI returns a list, try to group by document type mentioned in the string
                logger.warning("AI returned list instead of dictionary, attempting to group by document type")
                grouped_results = {}
                for issue in validation_results:
                    # Try to extract document type from the issue string
                    issue_lower = issue.lower()
                    doc_type = None
                    for doc in valid_documents:
                        doc_type_lower = doc.get("document_type", "").lower().replace(" ", "_")
                        if doc_type_lower in issue_lower:
                            doc_type = doc_type_lower
                            break
                    
                    if not doc_type:
                        # Default to first document type if can't determine
                        doc_type = valid_documents[0].get("document_type", "unknown").lower().replace(" ", "_")
                    
                    if doc_type not in grouped_results:
                        grouped_results[doc_type] = []
                    grouped_results[doc_type].append(issue)
                
                logger.info(f"Grouped {len(validation_results)} issues into {len(grouped_results)} document types")
                return grouped_results
            else:
                logger.warning("Validation results is not a dictionary or list, returning empty")
                return {}
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse validation result: {e}")
            if json_str:
                logger.error(f"Raw output: {json_str}")
            # Return empty dict on parse error instead of raising
            return {}
        except Exception as e:
            logger.error(f"Validation failed: {e}", exc_info=True)
            # Return empty dict on error instead of raising
            return {}

ai_validation_summary_processor = AIValidationSummary()