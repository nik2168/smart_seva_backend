# Smart Seva Verification API

A high-performance FastAPI server for document verification and processing using OCR, AI summarization, and LangGraph workflows. Supports multiple document types including images and PDFs with intelligent data extraction and validation.

## 🚀 Features

- **Multi-format OCR**: Supports images (JPEG, PNG, GIF, BMP, WebP) and PDF documents
- **Multiple OCR Engines**: Choose between PaddleOCR (fast, accurate), EasyOCR (lightweight), or Google Vision (cloud)
- **Intelligent OCR Fallback**: Automatic fallback to Google Vision API when primary OCR fails or has low confidence
- **Telugu Language Support**: Native support for Telugu (te) and English (en) OCR with dual-language processing
- **AI-Powered Summarization**: Uses DSPy and LangGraph for intelligent document analysis
- **Flexible LLM Providers**: Toggle between Ollama (local) and Groq (cloud) for AI processing
- **Batch Processing**: Process multiple documents in a single API call
- **Structured Data Extraction**: Automatically extracts and validates document fields
- **Cross-document Validation**: AI-powered validation across multiple documents
- **GPU Acceleration**: Optional GPU support for faster OCR processing
- **Comprehensive Logging**: Structured JSON logging for production monitoring

## 📋 Prerequisites

- **Python**: >= 3.13
- **uv**: Python package manager ([Installation guide](https://github.com/astral-sh/uv))
- **LLM Provider** (choose one):
  - **Ollama** (for local LLM): [Installation guide](https://ollama.ai)
    - Recommended model: `gpt-oss:20b-cloud` or similar
  - **Groq API** (for cloud LLM): [Get API key](https://console.groq.com)
- **Google Cloud Vision API** (optional, for OCR fallback): [Get API key](https://cloud.google.com/vision/docs/setup)
  - Required if using `google_vision` as primary OCR provider
  - Automatically used as fallback when primary OCR fails
- **GPU** (optional): CUDA-compatible GPU for faster OCR processing

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd smart_seva/server
```

### 2. Install dependencies

```bash
# Install all dependencies using uv
uv sync
```

This will install all required packages including:

- FastAPI and Uvicorn for the API server
- PaddleOCR/EasyOCR for OCR processing
- Google Cloud Vision API client (optional, for OCR fallback)
- DSPy for AI summarization
- LangGraph for workflow orchestration
- And other dependencies

### 3. Set up LLM Provider

Choose one of the following options:

#### Option A: Ollama (Local - Recommended for Development)

```bash
# Install Ollama (if not already installed)
# macOS: brew install ollama
# Linux: curl -fsSL https://ollama.ai/install.sh | sh

# Pull the recommended model
ollama pull gpt-oss:20b-cloud
```

#### Option B: Groq API (Cloud - Recommended for Production)

1. Sign up at [Groq Console](https://console.groq.com)
2. Generate an API key from your dashboard
3. Add the key to your `.env` file (see step 4)

### 3.5. Set up Google Vision API (Optional - for OCR fallback)

Google Vision API is automatically used as a fallback when primary OCR providers fail or have low confidence. It can also be used as the primary OCR provider.

1. Create a project in [Google Cloud Console](https://console.cloud.google.com)
2. Enable the Cloud Vision API
3. Create a service account and download the JSON key file
4. Add the key file path to your `.env` file (see step 4)

### 4. Configure environment variables

Create a `.env` file in the `server` directory:

```bash
# ============================================
# LLM Provider Configuration
# ============================================
# Choose provider: "ollama" or "groq"
LLM_PROVIDER=ollama

# Ollama Configuration (for LLM_PROVIDER=ollama)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gpt-oss:20b-cloud

# Groq API Configuration (for LLM_PROVIDER=groq)
# Get your API key from: https://console.groq.com
# GROQ_API_KEY=your_groq_api_key_here
# GROQ_MODEL=llama-3.1-70b-versatile

# ============================================
# OCR Configuration
# ============================================
OCR_PROVIDER=easyocr  # Options: "paddleocr", "easyocr", "google_vision"
OCR_LANGUAGES=["en", "te"]  # Languages: English (en) and Telugu (te) supported
OCR_GPU=true  # Enable GPU acceleration (if available)
OCR_LIGHTWEIGHT=true  # Use lightweight OCR models
OCR_MAX_IMAGE_DIMENSION=2000
OCR_MIN_CONFIDENCE=0.5  # Minimum confidence threshold for fallback (0.0-1.0)
OCR_BI_LANG=false  # If true, always run both Telugu and English OCR and combine results

# Google Vision API (optional - used as fallback automatically)
# Can be JSON file path or JSON string
# GOOGLE_VISION_API_KEY=/path/to/service-account-key.json
# Or: GOOGLE_VISION_API_KEY={"type": "service_account", ...}

# ============================================
# Logging
# ============================================
LOG_LEVEL=INFO
```

## 🚀 Quick Start

### Using Make (Recommended)

```bash
# Install dependencies
make install

# Start the API server
make api
```

### Manual Start

```bash
# Start the server
uv run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### Verify Installation

```bash
# Check health endpoint
curl http://localhost:8000/health

# Check API root
curl http://localhost:8000/
```

## 📚 API Documentation

Once the server is running, interactive API documentation is available at:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Main Endpoints

#### POST `/api/summarize`

Process and summarize multiple documents.

**Request:**

- `files`: List of files (multipart/form-data)
  - Supported formats: JPEG, PNG, GIF, BMP, WebP, PDF

**Response:**

```json
{
  "results": [
    {
      "file_name": "document.jpg",
      "success": true,
      "document_type": "aadhaar",
      "extracted_data": {
        "name": "John Doe",
        "aadhaar_number": "1234 5678 9012"
      },
      "ocr_text": "Full extracted text...",
      "ocr_confidence": 0.95,
      "errors": null
    }
  ],
  "validation_results": {
    "aadhaar": ["Issue 1", "Issue 2"]
  }
}
```

#### GET `/health`

Health check endpoint.

**Response:**

```json
{
  "status": "healthy"
}
```

## 🏗️ Project Structure

```
server/
├── src/
│   ├── api/
│   │   ├── main.py          # FastAPI application entry point
│   │   ├── routes.py        # API route handlers
│   │   └── models.py        # Pydantic models
│   ├── config/
│   │   ├── settings.py      # Application settings
│   │   └── logger_config.py # Logging configuration
│   ├── services/
│   │   ├── workflow.py      # LangGraph workflow definition
│   │   ├── states.py         # State models
│   │   ├── nodes/            # Workflow nodes
│   │   │   ├── input_validate_node.py
│   │   │   ├── image_ocr_node.py
│   │   │   └── ai_summary_node.py
│   │   └── processors/       # Document processors
│   │       ├── image_processor.py
│   │       └── ai_summary_processor.py
│   └── utils/
│       ├── ai_validation_summary.py
│       └── llm_config.py          # LLM provider configuration
├── docs/
│   ├── README.md
│   └── QUICK_START.md
├── test_scripts/
│   └── test_optimize_paddle_fast.py
├── pyproject.toml           # Project dependencies
├── Makefile                 # Convenience commands
└── README.md
```

## 🔧 Configuration

### LLM Provider Selection

The system supports multiple LLM providers for AI summarization and validation:

#### Ollama (Local - Default)

Best for development and offline use. Requires Ollama installed locally.

```bash
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gpt-oss:20b-cloud
```

#### Groq API (Cloud)

Best for production with fast inference. Requires API key.

```bash
LLM_PROVIDER=groq
GROQ_API_KEY=your_api_key_here
GROQ_MODEL=llama-3.1-70b-versatile
```

**Switching Providers**: Simply change `LLM_PROVIDER` in your `.env` file and restart the server. All AI processors will automatically use the new provider.

### OCR Provider Selection

The system supports multiple OCR engines with intelligent fallback:

1. **EasyOCR** (default): Lightweight, good for general use, supports Telugu
2. **PaddleOCR**: More accurate, better for complex documents, supports Telugu
3. **Google Vision**: Cloud-based, high accuracy, automatic fallback option

**Fallback Logic:**
- If primary OCR fails or confidence is below `OCR_MIN_CONFIDENCE`, the system automatically falls back to Google Vision API (if configured)
- This ensures high accuracy even when primary OCR struggles with complex documents

Change via environment variable:

```bash
OCR_PROVIDER=paddleocr  # or easyocr, google_vision
```

**Note**: Google Vision API is automatically used as fallback when available, even if not set as primary provider.

### GPU Configuration

For GPU acceleration (CUDA):

```bash
OCR_GPU=true
```

Ensure CUDA-compatible GPU drivers are installed.

### Language Support

The system natively supports **English (en)** and **Telugu (te)** languages for OCR processing.

Configure OCR languages in `.env`:

```bash
# Default: English and Telugu
OCR_LANGUAGES=["en", "te"]

# Enable dual-language processing (runs both languages and combines results)
OCR_BI_LANG=true
```

**Language Support Details:**
- **EasyOCR**: Supports both English and Telugu simultaneously
- **PaddleOCR**: Supports Telugu language model
- **Google Vision**: Automatically detects Telugu and other languages
- **Dual Language Mode**: When `OCR_BI_LANG=true`, runs both English and Telugu OCR and combines results for maximum accuracy

## 🧪 Testing

```bash
# Run tests using Make
make test

# Or manually
uv run python -m pytest test_scripts/
```

## 🛠️ Development

### Running in Development Mode

```bash
# Auto-reload on code changes
make api
```

### Code Quality

```bash
# Clean cache files
make clean
```

### Adding New Features

1. Add new nodes in `src/services/nodes/`
2. Update workflow in `src/services/workflow.py`
3. Add API routes in `src/api/routes.py`
4. Update models in `src/api/models.py`

## 📦 Dependencies

Key dependencies:

- **FastAPI**: Modern web framework
- **Uvicorn**: ASGI server
- **PaddleOCR/EasyOCR**: Primary OCR engines with Telugu support
- **Google Cloud Vision**: Cloud OCR with automatic fallback support
- **DSPy**: AI framework for LLM workflows with multi-provider support
- **LangGraph**: Workflow orchestration
- **Pydantic**: Data validation
- **Pillow**: Image processing
- **pdf2image**: PDF to image conversion

See `pyproject.toml` for complete dependency list.

## 🔄 Recent Updates

### LLM Provider Toggle

- Added support for multiple LLM providers (Ollama and Groq)
- Centralized LLM configuration in `src/utils/llm_config.py`
- Easy provider switching via environment variable
- Automatic configuration for all AI processors
- See `docs/LLM_CONFIGURATION.md` for detailed configuration guide

### OCR Enhancements

- **Google Vision API Fallback**: Automatic fallback to Google Vision when primary OCR fails or has low confidence
- **Telugu Language Support**: Native support for Telugu (te) OCR in both EasyOCR and PaddleOCR
- **Dual Language Processing**: Optional dual-language mode (`OCR_BI_LANG`) to run both English and Telugu OCR simultaneously
- **Intelligent Fallback**: System automatically uses Google Vision API as fallback when available, ensuring high accuracy

## 🐛 Troubleshooting

### Issue: OCR not working

**Solution**: 
- Ensure OCR models are downloaded. First run may take time to download models
- For Telugu support, ensure `OCR_LANGUAGES` includes `"te"`
- Check that Google Vision API key is configured if using fallback mode
- Verify primary OCR provider is correctly set in `.env`

### Issue: Telugu text not being recognized

**Solution**:
- Ensure `OCR_LANGUAGES=["en", "te"]` is set in `.env`
- Try enabling dual-language mode: `OCR_BI_LANG=true`
- Google Vision fallback automatically handles Telugu if primary OCR fails

### Issue: LLM provider connection error

**For Ollama**:
- Verify Ollama is running: `ollama list`
- Check `OLLAMA_BASE_URL` in `.env`
- Ensure model is pulled: `ollama pull gpt-oss:20b-cloud`

**For Groq**:
- Verify `GROQ_API_KEY` is set correctly in `.env`
- Check API key is valid at [Groq Console](https://console.groq.com)
- Ensure `LLM_PROVIDER=groq` is set
- Check internet connection (Groq requires cloud access)

**General**:
- Verify `LLM_PROVIDER` matches your configuration (either "ollama" or "groq")
- Check logs for specific error messages

### Issue: GPU not detected

**Solution**:

- Verify CUDA installation: `nvidia-smi`
- Set `OCR_GPU=false` to use CPU mode
- Check GPU drivers are up to date

### Issue: Import errors

**Solution**:

```bash
# Reinstall dependencies
uv sync --reinstall
```

## 📝 License

[Add your license here]

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact

[Add contact information or maintainer details]

---

**Note**: This project requires Python 3.13+ and uses `uv` for dependency management. Make sure you have the latest version of Python and `uv` installed.
