# LLM Provider Configuration Guide

This guide explains how to configure and toggle between different LLM providers (Ollama, Grok, Groq) for AI validation and summarization in Smart Seva.

## Overview

The application now supports multiple LLM providers that can be toggled via configuration settings. All AI validation and summarization tasks use the configured provider automatically.

## Supported Providers

1. **Ollama** (default) - Local LLM server
2. **Grok** - x.ai's Grok API
3. **Groq** - Groq API (alternative cloud provider)

## Configuration

### Environment Variables (.env file)

Create or update your `.env` file in the `server/` directory:

```bash
# LLM Provider Selection
# Options: "ollama", "grok", "groq"
LLM_PROVIDER=ollama

# ============================================
# Ollama Configuration (for LLM_PROVIDER=ollama)
# ============================================
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gpt-oss:20b-cloud

# ============================================
# Grok API Configuration (for LLM_PROVIDER=grok)
# ============================================
# Get your API key from: https://x.ai
GROK_API_KEY=your_grok_api_key_here
GROK_MODEL=grok-beta
GROK_BASE_URL=https://api.x.ai/v1

# ============================================
# Groq API Configuration (for LLM_PROVIDER=groq)
# ============================================
# Get your API key from: https://console.groq.com
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-70b-versatile
```

### Switching Providers

To switch between providers, simply change the `LLM_PROVIDER` environment variable:

**To use Ollama:**
```bash
LLM_PROVIDER=ollama
```

**To use Grok:**
```bash
LLM_PROVIDER=grok
GROK_API_KEY=your_api_key_here
```

**To use Groq:**
```bash
LLM_PROVIDER=groq
GROQ_API_KEY=your_api_key_here
```

## Setup Instructions

### 1. Ollama Setup (Default)

1. Install Ollama from [https://ollama.ai](https://ollama.ai)
2. Pull the recommended model:
   ```bash
   ollama pull gpt-oss:20b-cloud
   ```
3. Start Ollama server (usually runs automatically)
4. Set in `.env`:
   ```bash
   LLM_PROVIDER=ollama
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_MODEL=gpt-oss:20b-cloud
   ```

### 2. Grok API Setup

1. Sign up for Grok API access at [https://x.ai](https://x.ai)
2. Generate an API key from your account dashboard
3. Set in `.env`:
   ```bash
   LLM_PROVIDER=grok
   GROK_API_KEY=your_api_key_here
   GROK_MODEL=grok-beta
   GROK_BASE_URL=https://api.x.ai/v1
   ```

### 3. Groq API Setup

1. Sign up for Groq API access at [https://console.groq.com](https://console.groq.com)
2. Generate an API key from your account dashboard
3. Set in `.env`:
   ```bash
   LLM_PROVIDER=groq
   GROQ_API_KEY=your_api_key_here
   GROQ_MODEL=llama-3.1-70b-versatile
   ```

## How It Works

The application uses a centralized LLM configuration module (`src/utils/llm_config.py`) that:

1. Reads the `LLM_PROVIDER` setting from your configuration
2. Initializes DSPy with the appropriate provider
3. Automatically configures all AI processors to use the selected provider

### Code Flow

```
Settings (settings.py)
    ↓
LLM Config (llm_config.py)
    ↓
DSPy Configuration
    ↓
AI Processors (ai_validation_summary.py, ai_summary_processor.py)
```

## Usage Examples

### Programmatic Configuration

If you need to change providers programmatically:

```python
from src.config import settings
from src.utils.llm_config import configure_dspy, reset_dspy_config

# Change provider
settings.llm_provider = "grok"
settings.grok_api_key = "your_key"

# Reset and reconfigure
reset_dspy_config()
configure_dspy(force_reconfigure=True)
```

### Runtime Provider Switching

The configuration is loaded at module import time. To switch providers at runtime:

1. Update environment variables
2. Restart the application server
3. Or use the programmatic approach above

## Troubleshooting

### Common Issues

1. **"grok_api_key must be set"**
   - Ensure `GROK_API_KEY` is set in your `.env` file when using Grok

2. **"Unsupported LLM provider"**
   - Check that `LLM_PROVIDER` is one of: `ollama`, `grok`, `groq`

3. **Ollama connection errors**
   - Ensure Ollama is running: `ollama serve`
   - Check `OLLAMA_BASE_URL` matches your Ollama server address

4. **Grok API errors**
   - Verify your API key is valid
   - Check that `GROK_BASE_URL` is correct (default: `https://api.x.ai/v1`)

5. **DSPy not reconfiguring**
   - The configuration is cached. Use `force_reconfigure=True` or restart the server

## Files Modified

The following files were updated to support LLM provider toggling:

- `src/config/settings.py` - Added Grok/Groq configuration options
- `src/utils/llm_config.py` - New centralized LLM configuration module
- `src/utils/ai_validation_summary.py` - Updated to use centralized config
- `src/services/processors/ai_summary_processor.py` - Updated to use centralized config
- `old_processors/verification_processor.py` - Updated for consistency

## Best Practices

1. **Use environment variables** - Never hardcode API keys in source code
2. **Test configuration** - Verify your provider works before deploying
3. **Monitor usage** - Keep track of API usage for cloud providers (Grok/Groq)
4. **Fallback strategy** - Consider implementing fallback logic if primary provider fails
5. **Security** - Keep your `.env` file secure and never commit it to version control

## Cost Considerations

- **Ollama**: Free (runs locally)
- **Grok**: Check [x.ai pricing](https://x.ai) for current rates
- **Groq**: Check [Groq pricing](https://console.groq.com) for current rates

Choose the provider that best fits your needs, budget, and performance requirements.

