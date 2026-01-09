"""
LLM Configuration Module for DSPy

This module provides a centralized way to configure DSPy with different LLM providers
(Ollama, Grok, Groq) and allows easy toggling between them via settings.
"""
import dspy
import os
import logging
from src.config import settings

logger = logging.getLogger(__name__)

# Global variable to track if DSPy has been configured
_dspy_configured = False
_current_provider = None

def get_dspy_lm():
    """
    Initialize and return DSPy LM based on configured provider.
    Supports: ollama, grok, groq
    
    Returns:
        dspy.LM: Configured DSPy language model instance
        
    Raises:
        ValueError: If provider is unsupported or required config is missing
    """
    provider = settings.llm_provider.lower()
    
    if provider == "ollama":
        if not settings.ollama_model:
            raise ValueError("ollama_model must be set when using Ollama provider")
        lm = dspy.LM(
            f"ollama/{settings.ollama_model}",
            base_url=settings.ollama_base_url
        )
        logger.info(f"Initialized DSPy with Ollama: {settings.ollama_model} at {settings.ollama_base_url}")
        
    elif provider == "grok":
        if not settings.grok_api_key:
            raise ValueError("grok_api_key must be set when using Grok provider")
        if not settings.grok_model:
            raise ValueError("grok_model must be set when using Grok provider")
        
        # Grok API uses OpenAI-compatible endpoints
        # Set environment variable for DSPy to use
        os.environ["GROK_API_KEY"] = settings.grok_api_key
        
        # Use OpenAI format with Grok's base URL
        # DSPy's OpenAI LM supports custom base URLs via api_base parameter
        # Try both parameter names for compatibility
        try:
            lm = dspy.LM(
                model=f"openai/{settings.grok_model}",
                api_key=settings.grok_api_key,
                api_base=settings.grok_base_url
            )
        except TypeError:
            # Fallback: try with base_url parameter if api_base doesn't work
            lm = dspy.LM(
                model=f"openai/{settings.grok_model}",
                api_key=settings.grok_api_key,
                base_url=settings.grok_base_url
            )
        logger.info(f"Initialized DSPy with Grok: {settings.grok_model} at {settings.grok_base_url}")
        
    elif provider == "groq":
        if not settings.groq_api_key:
            raise ValueError("groq_api_key must be set when using Groq provider")
        if not settings.groq_model:
            raise ValueError("groq_model must be set when using Groq provider")
        
        # Set environment variable for DSPy
        os.environ["GROQ_API_KEY"] = settings.groq_api_key
        
        # DSPy has native Groq support
        lm = dspy.LM(
            f"groq/{settings.groq_model}",
            api_key=settings.groq_api_key
        )
        logger.info(f"Initialized DSPy with Groq: {settings.groq_model}")
        
    else:
        raise ValueError(
            f"Unsupported LLM provider: {provider}. "
            f"Supported providers: ollama, grok, groq"
        )
    
    return lm

def configure_dspy(force_reconfigure: bool = False):
    """
    Configure DSPy with the selected provider.
    This function is idempotent - it will only reconfigure if the provider changes
    or if force_reconfigure is True.
    
    Args:
        force_reconfigure: If True, reconfigure even if already configured
        
    Returns:
        dspy.LM: The configured language model instance
    """
    global _dspy_configured, _current_provider
    
    provider = settings.llm_provider.lower()
    
    # Check if we need to reconfigure
    if not force_reconfigure and _dspy_configured and _current_provider == provider:
        logger.debug(f"DSPy already configured with {provider}, skipping reconfiguration")
        return get_dspy_lm()
    
    # Get and configure the LM
    lm = get_dspy_lm()
    dspy.configure(lm=lm)
    
    _dspy_configured = True
    _current_provider = provider
    
    logger.info(f"DSPy successfully configured with provider: {provider}")
    return lm

def reset_dspy_config():
    """Reset DSPy configuration state. Useful for testing or switching providers."""
    global _dspy_configured, _current_provider
    _dspy_configured = False
    _current_provider = None
    logger.info("DSPy configuration state reset")

