# Image Forensics Quick Start Guide

## Installation

```bash
cd server
uv sync  # Installs all dependencies including forensics libraries
```

## Quick Test

```bash
# Test all forensic modules
uv run python -m src.services.nodes.image_forensics_node
```

## Basic Usage

### Option 1: Use Individual Modules

```python
from src.services.processors.image_forensics_processors import (
    ELAProcessor,
    NoiseProcessor,
    FontTextProcessor,
    MetadataProcessor,
    PrintScanProcessor
)

# Initialize
ela = ELAProcessor()
result = ela.analyze("document.jpg")

print(f"Score: {result['score']}")
print(f"Status: {result['status']}")
```

### Option 2: Use Complete Forensics Pipeline

```python
from src.services.nodes.image_forensics_node import image_forensics

# Create state
state = {
    "file_path": "document.jpg",
    "image_bytes": None,
    "form_fields": {},
    "ocr_text": "",
    # ... other fields
}

# Run all 5 modules
result = image_forensics(state)

# Get results
forensics = result["image_forensics"]
print(f"Overall Score: {forensics['overall_score']}")
print(f"Authentic: {forensics['summary']['is_authentic']}")
print(f"Concerns: {forensics['summary']['concerns']}")
```

### Option 3: Via Full Workflow

```python
from src.services.workflow import verification_graph

state = {
    "file_path": "document.jpg",
    "form_fields": {...},
    # ... required fields
}

result = verification_graph.invoke(state)
forensics = result["image_forensics"]
```

## Understanding Scores

| Score Range | Interpretation |
|-------------|---------------|
| 80-100 | ✅ Authentic |
| 60-79 | ⚠️ Minor concerns |
| 40-59 | 🔶 Moderate suspicion |
| 0-39 | ❌ High suspicion |

## Module-Specific Notes

### ELA (Error Level Analysis)
- **Best for**: Detecting copy-paste, splicing
- **Slow on**: Large images (>2MB)
- **Config**: Set `save_heatmap=False` in production

### Noise Pattern
- **Best for**: Compositing, region replacement
- **Options**: `use_wiener_filter=True` (fast), `use_nl_means=True` (slow, accurate)
- **Note**: Requires good quality images

### Font & Text
- **Best for**: Text manipulation, fake stamps
- **Requires**: At least 3 text regions
- **Note**: Works best on scanned documents

### Metadata
- **Best for**: Recent editing, tool detection
- **Fast**: 0.1-0.3s per image
- **Note**: Some images lack metadata

### Print-Scan
- **Best for**: Photocopied documents
- **Detects**: Halftone patterns from printing
- **Note**: May false-positive on low-quality scans

## Common Issues

### Issue: "No image available for forensics"
**Solution**: Ensure `file_path` or `image_bytes` is set in state

### Issue: Module errors but workflow continues
**Behavior**: Normal - failed modules are excluded from overall score

### Issue: Low scores on authentic documents
**Check**: Image quality, resolution, file format

### Issue: Slow processing
**Optimize**:
```python
# Disable expensive features
noise = NoiseProcessor(use_nl_means=False)
ela = ELAProcessor(save_heatmap=False)
```

## File Formats Supported

- **Images**: JPEG, PNG, TIFF, BMP
- **PDFs**: Metadata extraction only (not visual analysis)

## Next Steps

- Read full documentation: `docs/IMAGE_FORENSICS.md`
- Customize thresholds per your use case
- Integrate with your application workflow

