"""
Fraud Detection Modules

Pattern Analysis: Detects fraud patterns in extracted text JSON data
Error Level Analysis: ELA-based image forensics
Metadata Analysis: Metadata forensics for image files
"""

from .pattern_analysis import PatternAnalyzer
from .error_level_analysis import ELAProcessor
from .metadata_analysis import MetadataProcessor

__all__ = [
    "PatternAnalyzer",
    "ELAProcessor",
    "MetadataProcessor"
]
