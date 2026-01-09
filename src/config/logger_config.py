import logging
import sys
from src.config.settings import settings

# Get log level from settings, default to INFO if invalid
log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)