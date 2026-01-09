# SmartSeva Server Documentation

Welcome to the SmartSeva AI Document Verification documentation.

## Documentation Index

### 📘 [Image Forensics System](IMAGE_FORENSICS.md)
Complete technical documentation for the 5-module forensics system:
- Module 1: Error Level Analysis (ELA)
- Module 2: Noise Pattern Analysis
- Module 3: Font & Text Analysis
- Module 4: Metadata Forensics
- Module 5: Print-Scan Detection

**Topics Covered:**
- Architecture & workflow
- Technical implementation details
- Libraries & dependencies
- Configuration options
- Score interpretation
- Performance considerations
- API reference

### 🚀 [Quick Start Guide](QUICK_START.md)
Get started quickly with:
- Installation instructions
- Basic usage examples
- Common issues & solutions
- File format support
- Optimization tips

## System Overview

```
┌─────────────────────────────────────────────────────────┐
│                 SmartSeva Workflow                       │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  1. validate_input                                       │
│     ↓                                                     │
│  2. extract_ocr (PaddleOCR)                              │
│     ↓                                                     │
│  3. image_forensics (5 modules)                          │
│     ↓                                                     │
│  4. verify_document (DSPy + LLM)                         │
│     ↓                                                     │
│  5. END                                                   │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

## Key Features

### 🔍 Image Forensics
- **5 Specialized Modules** for comprehensive authenticity detection
- **Multi-layer Analysis** combining computer vision, signal processing, and metadata
- **Robust Scoring** with confidence levels and human-readable verdicts
- **Production Ready** with error handling and performance optimization

### 📝 OCR & Verification
- **PaddleOCR** for text extraction
- **DSPy + LLM** for intelligent document verification
- **Form Field Matching** with confidence scores
- **Document Type Detection**

### ⚡ Performance
- **Fast Processing**: 2-8 seconds per document
- **Scalable**: Handles multiple file formats
- **Configurable**: Tune for speed vs accuracy

## Dependencies

```toml
# Core
fastapi>=0.115.0
uvicorn[standard]>=0.32.0
langgraph>=1.0.4

# AI/ML
dspy-ai>=3.0.4
paddleocr>=3.3.2
paddlepaddle>=3.2.2

# Image Processing
opencv-python>=4.8.0
scikit-image>=0.25.2
pillow>=10.0.0

# Forensics
scipy>=1.11.0
pywavelets>=1.4.0
jpegio>=0.2.8
pypdf2>=3.0.0
```

## Project Structure

```
server/
├── src/
│   ├── api/              # FastAPI routes
│   ├── config/           # Configuration
│   ├── services/
│   │   ├── nodes/        # Workflow nodes
│   │   │   ├── image_forensics_node.py
│   │   │   ├── image_ocr_node.py
│   │   │   └── verification_node.py
│   │   ├── processors/   # Processing logic
│   │   │   └── image_forensics_processors/
│   │   │       ├── ela_processor.py
│   │   │       ├── noise_analysis_processor.py
│   │   │       ├── font_text_processor.py
│   │   │       ├── metadata_processor.py
│   │   │       └── print_scan_processor.py
│   │   ├── states.py     # State definitions
│   │   └── workflow.py   # LangGraph workflow
│   └── test_data/        # Test images
├── docs/                 # This directory
└── pyproject.toml        # Dependencies
```

## Testing

```bash
# Test individual modules
uv run python -m src.services.processors.image_forensics_processors.ela_processor
uv run python -m src.services.processors.image_forensics_processors.noise_analysis_processor

# Test complete forensics pipeline
uv run python -m src.services.nodes.image_forensics_node

# Run API server
uv run uvicorn src.api.main:app --reload
```

## API Usage

```bash
# Verify a document
curl -X POST "http://localhost:8000/verify" \
  -F "file=@document.jpg" \
  -F "form_fields={\"name\": \"John Doe\"}"
```

Response includes:
- OCR extracted text
- Document type
- Form verification results
- **Image forensics scores** (5 modules + overall)
- Authenticity summary

## Support

For detailed information on specific modules or features, refer to:
- [IMAGE_FORENSICS.md](IMAGE_FORENSICS.md) - Complete forensics documentation
- [QUICK_START.md](QUICK_START.md) - Getting started guide

## Version

Current Version: 0.1.0
Last Updated: December 2024

