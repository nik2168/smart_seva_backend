"""
pattern_analysis.py

Pattern Analysis Engine for fraud detection on extracted text JSON data.
Implements easy-to-implement operations:
1. Temporal Pattern Analysis - Date validations and chronological checks
2. Statistical Outlier Detection - Unrealistic value detection
3. Simple Entity Pattern Analysis - Duplicate detection and suspicious patterns

Returns:
{
 "module": "PatternAnalysis",
 "score": float(0-100),
 "details": { ... analysis results ... },
 "status": "success" / "error"
}
"""

import json
import re
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dateutil import parser as date_parser
from dateutil.relativedelta import relativedelta

# Indian national holidays (common ones - can be expanded)
INDIAN_HOLIDAYS = [
    "2024-01-26", "2024-08-15", "2024-10-02", "2024-10-31", "2024-11-01",
    "2025-01-26", "2025-08-15", "2025-10-02", "2025-10-31", "2025-11-01",
    # Add more as needed
]

class PatternAnalyzer:
    """
    Pattern Analysis Engine for detecting fraud patterns in extracted document data.
    """
    
    def __init__(self):
        self.score = 0
        self.flags = []
        self.details = {}
    
    def analyze(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze extracted JSON data for fraud patterns.
        
        Args:
            extracted_data: Dictionary containing extracted fields from OCR
            
        Returns:
            Dictionary with analysis results including score (0-100)
        """
        try:
            if not extracted_data:
                return {
                    "module": "PatternAnalysis",
                    "score": 0,
                    "status": "error",
                    "error_msg": "No extracted data provided"
                }
            
            self.score = 0
            self.flags = []
            self.details = {
                "temporal_analysis": {},
                "statistical_analysis": {},
                "entity_analysis": {}
            }
            
            # 1. Temporal Pattern Analysis
            self._temporal_analysis(extracted_data)
            
            # 2. Statistical Outlier Detection
            self._statistical_analysis(extracted_data)
            
            # 3. Simple Entity Pattern Analysis
            self._entity_analysis(extracted_data)
            
            # Cap score at 100
            self.score = min(100, self.score)
            
            return {
                "module": "PatternAnalysis",
                "score": self.score,
                "status": "success",
                "details": {
                    **self.details,
                    "total_flags": len(self.flags),
                    "flags": self.flags
                }
            }
            
        except Exception as e:
            return {
                "module": "PatternAnalysis",
                "score": 0,
                "status": "error",
                "error_msg": str(e)
            }
    
    def _temporal_analysis(self, data: Dict[str, Any]):
        """Temporal Pattern Analysis - Date validations and chronological checks"""
        temporal_flags = []
        
        # Extract date fields
        date_fields = {}
        for key, value in data.items():
            if 'date' in key.lower() and value:
                date_fields[key] = value
        
        # Check each date
        for field_name, date_str in date_fields.items():
            parsed_date = self._parse_date(date_str)
            if not parsed_date:
                continue
            
            # Check 1: Weekend issuance
            if field_name in ['date_of_issue', 'date_of_registration', 'issue_date']:
                if parsed_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
                    self.score += 15
                    flag = f"Certificate issued on weekend ({parsed_date.strftime('%A')})"
                    temporal_flags.append(flag)
                    self.flags.append(flag)
            
            # Check 2: Holiday issuance
            date_str_iso = parsed_date.strftime("%Y-%m-%d")
            if date_str_iso in INDIAN_HOLIDAYS:
                self.score += 20
                flag = f"Certificate issued on national holiday ({date_str_iso})"
                temporal_flags.append(flag)
                self.flags.append(flag)
            
            # Check 3: Future dates
            if parsed_date > datetime.now():
                self.score += 25
                flag = f"Future date detected in {field_name}: {date_str}"
                temporal_flags.append(flag)
                self.flags.append(flag)
            
            # Check 4: Very old dates (before 1900)
            if parsed_date.year < 1900:
                self.score += 20
                flag = f"Very old date in {field_name}: {date_str}"
                temporal_flags.append(flag)
                self.flags.append(flag)
        
        # Chronological logic checks
        birth_date = self._parse_date(data.get('date_of_birth') or data.get('birth_date'))
        registration_date = self._parse_date(data.get('date_of_registration') or data.get('registration_date'))
        issue_date = self._parse_date(data.get('date_of_issue') or data.get('issue_date'))
        marriage_date = self._parse_date(data.get('date_of_marriage') or data.get('marriage_date'))
        
        # Check: Registration date should be after birth date
        if birth_date and registration_date:
            if registration_date < birth_date:
                self.score += 30
                flag = "Registration date is before birth date (chronological inconsistency)"
                temporal_flags.append(flag)
                self.flags.append(flag)
        
        # Check: Issue date should be after registration date
        if registration_date and issue_date:
            if issue_date < registration_date:
                self.score += 25
                flag = "Issue date is before registration date (chronological inconsistency)"
                temporal_flags.append(flag)
                self.flags.append(flag)
        
        # Check: Marriage date should be after birth date
        if birth_date and marriage_date:
            age_at_marriage = relativedelta(marriage_date, birth_date).years
            if age_at_marriage < 18:
                self.score += 35
                flag = f"Marriage date indicates age below 18 ({age_at_marriage} years)"
                temporal_flags.append(flag)
                self.flags.append(flag)
            elif marriage_date < birth_date:
                self.score += 40
                flag = "Marriage date is before birth date (impossible)"
                temporal_flags.append(flag)
                self.flags.append(flag)
        
        # Age-date consistency check
        age = self._extract_age(data)
        if birth_date and age is not None:
            calculated_age = relativedelta(datetime.now(), birth_date).years
            age_diff = abs(calculated_age - age)
            if age_diff > 2:  # Allow 2 years tolerance
                self.score += 20
                flag = f"Age mismatch: stated age {age} vs calculated age {calculated_age} from birth date"
                temporal_flags.append(flag)
                self.flags.append(flag)
        
        self.details["temporal_analysis"] = {
            "flags": temporal_flags,
            "date_fields_found": list(date_fields.keys())
        }
    
    def _statistical_analysis(self, data: Dict[str, Any]):
        """Statistical Outlier Detection - Unrealistic values"""
        statistical_flags = []
        
        # Check income (if present)
        income = self._extract_numeric(data.get('income') or data.get('annual_income') or data.get('monthly_income'))
        if income:
            # Unrealistically high income (> 1 crore)
            if income > 10000000:  # 1 crore
                self.score += 30
                flag = f"Unrealistically high income: ₹{income:,}"
                statistical_flags.append(flag)
                self.flags.append(flag)
            # Unrealistically low income (< 1000)
            elif income < 1000:
                self.score += 15
                flag = f"Unrealistically low income: ₹{income:,}"
                statistical_flags.append(flag)
                self.flags.append(flag)
        
        # Check age
        age = self._extract_age(data)
        if age is not None:
            if age > 120:
                self.score += 25
                flag = f"Unrealistically high age: {age} years"
                statistical_flags.append(flag)
                self.flags.append(flag)
            elif age < 0:
                self.score += 30
                flag = f"Invalid negative age: {age}"
                statistical_flags.append(flag)
                self.flags.append(flag)
        
        # Check family size
        family_size = self._extract_numeric(data.get('family_size') or data.get('number_of_family_members'))
        if family_size:
            if family_size > 50:
                self.score += 20
                flag = f"Unrealistically large family size: {family_size}"
                statistical_flags.append(flag)
                self.flags.append(flag)
            elif family_size < 1:
                self.score += 15
                flag = f"Invalid family size: {family_size}"
                statistical_flags.append(flag)
                self.flags.append(flag)
        
        # Check Aadhaar number format (if present)
        aadhaar = data.get('aadhaar_number') or data.get('aadhaar') or data.get('uid')
        if aadhaar:
            aadhaar_str = str(aadhaar).replace(' ', '').replace('-', '')
            if len(aadhaar_str) != 12 or not aadhaar_str.isdigit():
                self.score += 25
                flag = f"Invalid Aadhaar number format: {aadhaar}"
                statistical_flags.append(flag)
                self.flags.append(flag)
        
        # Check PAN number format (if present)
        pan = data.get('pan_number') or data.get('pan')
        if pan:
            pan_str = str(pan).upper().replace(' ', '')
            pan_pattern = r'^[A-Z]{5}[0-9]{4}[A-Z]{1}$'
            if not re.match(pan_pattern, pan_str):
                self.score += 20
                flag = f"Invalid PAN number format: {pan}"
                statistical_flags.append(flag)
                self.flags.append(flag)
        
        self.details["statistical_analysis"] = {
            "flags": statistical_flags
        }
    
    def _entity_analysis(self, data: Dict[str, Any]):
        """Simple Entity Pattern Analysis - Duplicates and suspicious patterns"""
        entity_flags = []
        
        # Check for duplicate values across different fields
        field_values = {}
        for key, value in data.items():
            if value and isinstance(value, (str, int, float)):
                value_str = str(value).strip()
                if value_str and len(value_str) > 3:  # Only check meaningful values
                    if value_str not in field_values:
                        field_values[value_str] = []
                    field_values[value_str].append(key)
        
        # Find duplicates
        for value, fields in field_values.items():
            if len(fields) > 1:
                # Same value in multiple fields (could be suspicious)
                if any('name' in f.lower() for f in fields) and any('address' in f.lower() for f in fields):
                    self.score += 15
                    flag = f"Same value '{value[:30]}...' appears in multiple fields: {', '.join(fields)}"
                    entity_flags.append(flag)
                    self.flags.append(flag)
        
        # Check for suspicious text patterns
        for key, value in data.items():
            if isinstance(value, str):
                value_str = str(value).strip()
                
                # All caps (might indicate fake/copied data)
                if value_str.isupper() and len(value_str) > 10:
                    if key.lower() not in ['registration_number', 'certificate_number', 'id']:
                        self.score += 5
                        flag = f"All caps text in {key} (suspicious pattern)"
                        entity_flags.append(flag)
                        self.flags.append(flag)
                
                # Repeated characters (e.g., "AAAAA", "11111")
                if len(value_str) > 5:
                    if len(set(value_str)) < 3:  # Less than 3 unique characters
                        self.score += 10
                        flag = f"Repeated character pattern in {key}: {value_str[:20]}..."
                        entity_flags.append(flag)
                        self.flags.append(flag)
                
                # Suspiciously short or generic values
                if key in ['name', 'father_name', 'mother_name'] and len(value_str) < 3:
                    self.score += 15
                    flag = f"Suspiciously short {key}: '{value_str}'"
                    entity_flags.append(flag)
                    self.flags.append(flag)
        
        # Check for placeholder/test data
        test_patterns = ['test', 'sample', 'example', 'dummy', 'xxxx', 'yyyy', 'zzzz']
        for key, value in data.items():
            if isinstance(value, str):
                value_lower = value.lower()
                if any(pattern in value_lower for pattern in test_patterns):
                    self.score += 20
                    flag = f"Test/placeholder data detected in {key}: {value}"
                    entity_flags.append(flag)
                    self.flags.append(flag)
        
        self.details["entity_analysis"] = {
            "flags": entity_flags
        }
    
    def _parse_date(self, date_str: Any) -> Optional[datetime]:
        """Parse date string to datetime object"""
        if not date_str:
            return None
        
        try:
            # Try parsing with dateutil (handles various formats)
            return date_parser.parse(str(date_str), dayfirst=True)
        except:
            try:
                # Try common formats
                formats = [
                    "%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d",
                    "%d-%B-%Y", "%d %B %Y", "%B %d, %Y",
                    "%d-%b-%Y", "%d %b %Y", "%b %d, %Y"
                ]
                for fmt in formats:
                    try:
                        return datetime.strptime(str(date_str), fmt)
                    except:
                        continue
            except:
                pass
        
        return None
    
    def _extract_age(self, data: Dict[str, Any]) -> Optional[int]:
        """Extract age from data"""
        age_str = data.get('age') or data.get('years')
        if age_str:
            try:
                return int(float(str(age_str)))
            except:
                pass
        return None
    
    def _extract_numeric(self, value: Any) -> Optional[float]:
        """Extract numeric value from string"""
        if value is None:
            return None
        
        if isinstance(value, (int, float)):
            return float(value)
        
        if isinstance(value, str):
            # Remove currency symbols and commas
            cleaned = re.sub(r'[₹,\s]', '', value)
            try:
                return float(cleaned)
            except:
                pass
        
        return None


if __name__ == "__main__":
    # Example Usage
    analyzer = PatternAnalyzer()
    
    # Test with sample data
    test_data = {
        "name": "Mohammad Husain",
        "gender": "Male",
        "date_of_birth": "16-January-2017",
        "place_of_birth": "Gava",
        "father_name": "Shafique Ahmed",
        "mother_name": "Farzana Aslam",
        "permanent_address": "Aliganj; Road No-3 City-Gava, Dist Gava, State-Bihar;",
        "registration_number": "12559/1/7428/2017",
        "date_of_registration": "19-June-2017",
        "date_of_issue": "19-June-2017",
        "signature_of_issuing_authority": "36879/0"
    }
    result = analyzer.analyze(test_data)
    print(json.dumps(result, indent=2))

# uv run python src/services/modules/fraud_detection/pattern_analysis.py    