from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    # API Server Configuration
    api_host: str = "0.0.0.0"
    api_port: int = int(os.getenv("PORT", "8000"))  # Render.io provides PORT env var
    api_reload: bool = False  # Disable reload in production
    
    # LLM Provider Configuration
    llm_provider: str = "ollama"  # Options: "ollama", "grok", "groq" (default: ollama)
    
    # Ollama Configuration
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "gpt-oss:20b-cloud"
    
    # Groq API Configuration (alternative)
    groq_api_key: str | None = None  # Groq API key
    groq_model: str = "llama-3.1-70b-versatile"  # Default Groq model
    
    # OCR Configuration
    ocr_provider: str = "easyocr"  # Options: "paddleocr", "easyocr", "google_vision", or "cloud" (default: easyocr)
    ocr_languages: list[str] = ["en", "te"]
    ocr_gpu: bool = True
    ocr_lightweight: bool = True
    ocr_max_image_dimension: int = 2000
    ocr_min_confidence: float = 0.5  # Minimum confidence threshold for fallback (0.0-1.0)
    ocr_bi_lang: bool = False  # If True, always run both Telugu and English OCR and combine results
    google_vision_api_key: str | None = None  # Google Cloud Vision API key (required if ocr_provider=google_vision)
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        # Optionally allow extra fields if you don't want to define all
        # extra = "ignore"  # Uncomment this if you want to ignore extra fields

settings = Settings()