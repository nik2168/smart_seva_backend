import logging  # Standard logging utilities
import tempfile  # Provides temporary file helpers
import os  # OS-level path and file helpers
import io  # In-memory byte streams
from PIL import Image  # Image handling
import numpy as np  # Numerical operations for image analysis
from src.config import settings  # Application settings shared across modules
from time import time

logger = logging.getLogger(__name__)  # Module-level logger instance

# Conditional imports based on provider
try:  # Attempt to import PaddleOCR for OCR processing
    from paddleocr import PaddleOCR
    import paddle
    PADDLEOCR_AVAILABLE = True  # Flag indicating PaddleOCR is ready
except ImportError:  # Handle missing PaddleOCR dependency
    PADDLEOCR_AVAILABLE = False  # Flag off when PaddleOCR not installed
    paddle = None  # Placeholder to avoid NameError later
    logger.warning("PaddleOCR not available")  # Inform about missing provider

try:  # Attempt to import EasyOCR as alternative provider
    import easyocr
    EASYOCR_AVAILABLE = True  # Flag indicating EasyOCR is ready
except ImportError:  # Handle missing EasyOCR dependency
    EASYOCR_AVAILABLE = False  # Flag off when EasyOCR not installed
    logger.warning("EasyOCR not available")  # Inform about missing provider


class OCRService:
    _instance = None  # Holds singleton instance
    _paddleocr = None  # Cached PaddleOCR instance
    _easyocr = None    # Cached EasyOCR instance
    _provider = None  # Cached provider selection
    
    def __new__(cls):
        if cls._instance is None:  # Create singleton lazily
            cls._instance = super().__new__(cls)  # Instantiate base object
        return cls._instance  # Return shared instance
    
    def _get_provider(self):
        """Get the OCR provider from settings"""
        if self._provider is None:  # Resolve provider once
            provider = settings.ocr_provider.lower()  # Normalize configured provider
            if provider is None:
                provider = "paddleocr"
            else:
                self._provider = provider
        return self._provider  # Return cached provider
    
    def _get_paddleocr(self):
        """Initialize PaddleOCR with optimized low-memory configuration"""
        if self._paddleocr is None:  # Lazily build PaddleOCR
            logger.info("Initializing PaddleOCR with English language (optimized for memory)")
            
            # Configure PaddlePaddle for memory optimization
            if paddle is not None:  # Only when paddle is available
                try:
                    paddle.set_flags({
                        "FLAGS_fraction_of_cpu_memory_to_use": 0.85,  # Cap RAM usage
                        "FLAGS_use_pinned_memory": False,  # Avoid pinned memory on macOS
                    })
                    logger.debug("Paddle memory flags set for low-RAM environment")
                except Exception as e:  # Non-fatal flag failures
                    logger.warning(f"Paddle flags could not be applied: {e}")
            
            lang = 'te'  # Default language for PaddleOCR
            if lang not in settings.ocr_languages:  # Warn if Telugu requested
                logger.warning(f"{lang} OCR via PaddleOCR not supported; using English model")
            
            try:
                self._paddleocr = PaddleOCR(
                    use_textline_orientation=True, # Enable textline orientation detection
                    lang=lang,  # Language selection
                    cpu_threads=2,  # Limit CPU threads to reduce memory usage
                    enable_mkldnn=False,  # Disable MKLDNN on Apple Silicon
                )
                logger.info("PaddleOCR initialized successfully with optimized configuration")
            except Exception as e:  # Surface initialization errors
                logger.error(f"PaddleOCR initialization failed: {e}")
                raise
        return self._paddleocr  # Return cached PaddleOCR instance
    
    def _get_easyocr(self):
        """Initialize EasyOCR reader"""
        if self._easyocr is None:  # Lazily build EasyOCR
            logger.info("Initializing EasyOCR with English language")
            # EasyOCR language codes: 'en' for English
            lang = 'en'  # Default language code
            if 'te' in settings.ocr_languages:  # Include Telugu when configured
                lang = ['en', 'te']  # EasyOCR supports multiple languages
                logger.info("EasyOCR initialized with English and Telugu")
            else:
                lang = ['en']  # Keep English only
            
            self._easyocr = easyocr.Reader(lang, gpu=settings.ocr_gpu)  # Build reader
            logger.info("EasyOCR initialized successfully")
        return self._easyocr  # Return cached EasyOCR instance
    
    def _resize_image_if_needed(self, image_bytes: bytes, max_dimension: int = None) -> bytes:
        """Resize very large images to cap memory usage."""
        if max_dimension is None:  # Use configured max when not provided
            max_dimension = min(settings.ocr_max_image_dimension, 2400)  # Clamp to 2400px
        
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")  # Load image bytes
            width, height = img.size  # Current dimensions
            
            if max(width, height) > max_dimension:  # Only resize large images
                scale = max_dimension / max(width, height)  # Compute scale factor
                new_size = (int(width * scale), int(height * scale))  # Target size
                img = img.resize(new_size, Image.Resampling.LANCZOS)  # High-quality resize
                logger.info(f"Downscaled image from {width}x{height} to {new_size}")
                
                output = io.BytesIO()  # Buffer for JPEG output
                img.save(output, format="JPEG", quality=85, optimize=True)  # Save compressed
                return output.getvalue()  # Return resized bytes
            
            return image_bytes  # Return original if already small
        except Exception as e:  # On any failure, keep original
            logger.warning(f"Image resize failed, using original image: {e}")
            return image_bytes
    
    def _find_safe_split_point(self, img: Image.Image) -> int:
        """
        Find a safe horizontal split point by detecting vertical whitespace/gaps.
        Returns the x-coordinate where there's minimal text content (safe to split).
        """
        width, height = img.size  # Image dimensions
        gray = img.convert("L")  # Convert to grayscale for intensity analysis
        arr = np.asarray(gray, dtype=np.float32)  # Pixel data as float array
        
        # Calculate vertical projection: sum of pixel values for each column
        # Lower values indicate more whitespace/background
        vertical_projection = np.sum(arr, axis=0)  # Column intensity totals
        
        # Normalize to 0-1 range
        normalized = (vertical_projection - vertical_projection.min()) / (vertical_projection.max() - vertical_projection.min() + 1e-6)  # Avoid division by zero
        
        # Look for split point around the middle (40-60% of width)
        search_start = int(width * 0.4)  # Start of search window
        search_end = int(width * 0.6)  # End of search window
        
        # Find column with minimum text content (maximum whitespace) in the middle region
        search_region = normalized[search_start:search_end]  # Middle slice for search
        min_idx = np.argmin(search_region)  # Index of lowest intensity (most blank)
        split_point = search_start + min_idx  # Translate to absolute x-coordinate
        
        logger.debug(f"Found safe split point at x={split_point} (whitespace score: {normalized[split_point]:.3f})")
        return split_point  # Return safe split x-position
    
    def _split_image_into_tiles(self, img: Image.Image):
        """
        Split image only if necessary. Tries to find a safe split point at whitespace/gaps
        to avoid cutting through text. Returns list with single image if no split needed.
        """
        width, height = img.size  # Current image dimensions
        max_safe_width = 2000  # Process whole image if width <= 2000px
        
        # If image is small enough, process whole without splitting
        if width <= max_safe_width:
            logger.debug(f"Image width {width}px is within safe limit, processing whole image")
            return [img]  # Return single tile (no split)
        
        # Image is too large, need to split - find safe split point
        logger.debug(f"Image width {width}px exceeds safe limit, finding safe split point")
        split_point = self._find_safe_split_point(img)  # Locate whitespace split
        
        # Add overlap to ensure we don't miss text at boundaries
        overlap = int(width * 0.1)  # 10% overlap for safety
        
        # Left tile: from start to split_point + overlap
        left_tile = img.crop((0, 0, min(split_point + overlap, width), height))
        
        # Right tile: from split_point - overlap to end
        right_tile = img.crop((max(0, split_point - overlap), 0, width, height))
        
        tiles = [left_tile, right_tile]  # Collect resulting tiles
        
        logger.debug(
            f"Image split at safe point x={split_point} with {overlap}px overlap "
            f"(left: {left_tile.size[0]}x{height}, right: {right_tile.size[0]}x{height})"
        )
        
        return tiles  # Return list of tiles
    
    def _is_informative_tile(self, tile: Image.Image, min_std: float = 6.0) -> bool:
        """Skip tiles that are mostly blank/flat to save time."""
        try:
            arr = np.asarray(tile.convert("L"), dtype=np.float32)  # Grayscale array
            return arr.std() >= min_std  # True if variance indicates content
        except Exception as e:  # On errors, prefer processing tile
            logger.warning(f"Error checking tile informativeness: {e}")
            return True  # Default to processing if check fails
    
    def _extract_text_paddleocr(self, image_bytes: bytes) -> tuple[str, float]:
        """
        Extract text using PaddleOCR. Processes whole image if small enough,
        otherwise splits at safe whitespace boundaries to avoid cutting text.
        Returns (text, average_confidence)
        """
        # Step 1: Resize overly large images
        image_bytes = self._resize_image_if_needed(image_bytes)  # Downscale if needed
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")  # Load image for tiling
        
        # Step 2: Get tiles (may be single whole image or split at safe boundary)
        tiles = self._split_image_into_tiles(img)  # Split when wide
        
        ocr = self._get_paddleocr()  # Acquire PaddleOCR instance
        all_texts = []  # Collected text blocks
        all_confidences = []  # Collected confidences
        
        # Reuse a single temp file for all tiles to reduce FS overhead
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp_path = tmp.name  # Path to shared temp file
        
        try:
            for idx, tile in enumerate(tiles):  # Process each tile
                # Skip blank tiles
                if not self._is_informative_tile(tile):  # Skip low-variance tiles
                    logger.debug(f"Skipping tile {idx}: low variance (likely blank)")
                    continue
                
                # Save tile to temp file
                tile.save(tmp_path, format="JPEG", quality=85)  # Persist tile for OCR
                
                try:
                    result = ocr.ocr(tmp_path)  # Run PaddleOCR on tile
                    if result and isinstance(result, list):  # Validate response structure
                        block = result[0]  # PaddleOCR returns list with first block
                        rec_texts = block.get("rec_texts", [])  # Extract texts
                        rec_scores = block.get("rec_scores", [])  # Extract confidences
                        
                        for text, score in zip(rec_texts, rec_scores):  # Pair text/score
                            if text and isinstance(text, str):  # Validate text
                                text = text.strip()  # Trim whitespace
                                confidence = float(score) if score is not None else 1.0  # Normalize score
                                if text and confidence >= 0.3:  # Keep confident hits
                                    all_texts.append(text)  # Store text
                                    all_confidences.append(confidence)  # Store confidence
                except Exception as e:  # Handle per-tile OCR errors
                    logger.warning(f"OCR failed on tile {idx}: {e}")
        finally:
            os.unlink(tmp_path)  # Ensure temp file cleanup
        
        # Deduplicate results while preserving order
        unique_texts = list(dict.fromkeys(all_texts))  # Remove duplicates in order
        
        # Calculate average confidence from all confidences (before deduplication)
        avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0  # Mean confidence
        
        logger.debug(
            f"PaddleOCR extracted {len(unique_texts)} unique text blocks "
            f"from {len(tiles)} parts"
        )
        
        return "\n".join(unique_texts), avg_confidence  # Return combined text and average score
    
    def _extract_text_easyocr(self, image_bytes: bytes) -> tuple[str, float]:
        """Extract text using EasyOCR, returns (text, average_confidence)"""
        try:
            reader = self._get_easyocr()  # Acquire EasyOCR reader
            # EasyOCR accepts: file path (string), bytes, or numpy array
            # Convert PIL Image to numpy array for EasyOCR
            image = Image.open(io.BytesIO(image_bytes))  # Load image from bytes
            image_np = np.array(image)  # Convert to numpy array
            result = reader.readtext(image_np)  # Run OCR
            
            texts = []  # Collected text snippets
            confidences = []  # Collected confidence scores
            for detection in result:  # Iterate detections
                text = detection[1]  # Text is at index 1
                confidence = detection[2]  # Confidence is at index 2
                
                if text and isinstance(text, str):  # Validate text
                    text = text.strip()  # Trim whitespace
                    if text and confidence >= 0.3:  # Keep confident hits
                        texts.append(text)  # Store text
                        confidences.append(confidence)  # Store confidence
            
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0  # Mean confidence
            return "\n".join(texts), avg_confidence  # Return combined result
        except Exception as e:  # Propagate errors with context
            logger.error(f"EasyOCR extraction failed: {e}", exc_info=True)
            raise
    
    def extract_text(self, image_bytes: bytes) -> tuple[str, float]:
        """
        Extract text from image bytes with fallback logic:
        - Try EasyOCR first (or configured provider)
        - If confidence is too low, fallback to PaddleOCR
        Returns: (text, average_confidence)
        """
        # Determine primary provider (default to EasyOCR)
        primary_provider = settings.ocr_provider.lower()  # Read configured provider
        if primary_provider not in ["easyocr", "paddleocr"]:  # Validate choice
            primary_provider = "easyocr"  # Default to EasyOCR
        
        # Try primary provider first
        try:
            if primary_provider == "easyocr" and EASYOCR_AVAILABLE:  # Use EasyOCR path
                logger.info("Attempting OCR extraction with EasyOCR")
                text, confidence = self._extract_text_easyocr(image_bytes)  # Run EasyOCR
                logger.info(f"EasyOCR completed with average confidence: {confidence:.2f}")
                
                # Check if confidence is too low and fallback is available
                if confidence < settings.ocr_min_confidence and PADDLEOCR_AVAILABLE:
                    logger.warning(f"EasyOCR confidence ({confidence:.2f}) below threshold ({settings.ocr_min_confidence}), trying PaddleOCR fallback")
                    try:
                        text_paddle, confidence_paddle = self._extract_text_paddleocr(image_bytes)  # Fallback OCR
                        logger.info(f"PaddleOCR fallback completed with average confidence: {confidence_paddle:.2f}")
                        # Use PaddleOCR result if it's better or if EasyOCR was too low
                        if confidence_paddle > confidence:
                            logger.info("Using PaddleOCR result (higher confidence)")
                            return text_paddle, confidence_paddle  # Prefer better result
                        else:
                            logger.info("Keeping EasyOCR result despite low confidence")
                            return text, confidence  # Retain original
                    except Exception as e:  # Fallback failure handling
                        logger.warning(f"PaddleOCR fallback failed: {e}, using EasyOCR result")
                        return text, confidence  # Return EasyOCR output
                
                return text, confidence  # Successful EasyOCR path
            
            elif primary_provider == "paddleocr" and PADDLEOCR_AVAILABLE:  # Use PaddleOCR path
                logger.info("Attempting OCR extraction with PaddleOCR")
                text, confidence = self._extract_text_paddleocr(image_bytes)  # Run PaddleOCR
                logger.info(f"PaddleOCR completed with average confidence: {confidence:.2f}")
                # No fallback for PaddleOCR - return result directly
                return text, confidence
            
            else:
                # Fallback to available provider
                if EASYOCR_AVAILABLE:  # Use EasyOCR if available
                    logger.warning(f"{primary_provider} not available, using EasyOCR")
                    return self._extract_text_easyocr(image_bytes)
                elif PADDLEOCR_AVAILABLE:  # Use PaddleOCR if available
                    logger.warning(f"{primary_provider} not available, using PaddleOCR")
                    return self._extract_text_paddleocr(image_bytes)
                else:
                    raise RuntimeError("No OCR provider available")  # Neither provider present
        
        except Exception as e:  # Handle primary provider failures
            logger.error(f"OCR extraction failed with primary provider: {e}", exc_info=True)
            # Only fallback from EasyOCR to PaddleOCR (not the reverse)
            if primary_provider == "easyocr" and PADDLEOCR_AVAILABLE:
                logger.info("Trying PaddleOCR as fallback")
                try:
                    text, confidence = self._extract_text_paddleocr(image_bytes)  # Attempt fallback
                    # Check if fallback confidence is below threshold
                    if confidence < settings.ocr_min_confidence:
                        logger.warning(f"PaddleOCR fallback confidence ({confidence:.2f}) below threshold ({settings.ocr_min_confidence}), but returning result as fallback")
                    return text, confidence  # Return fallback output regardless
                except Exception as fallback_error:  # If fallback also fails
                    logger.error(f"PaddleOCR fallback also failed: {fallback_error}")
                    raise  # Propagate failure
            # If PaddleOCR fails, don't fallback - just raise the error
            raise


ocr_service = OCRService()  # Singleton instance for external use


if __name__ == "__main__":
    ocr_service = OCRService()
    # Assume running from server/ directory
    test_image_path = "test_data/telgu.jpeg"
    with open(test_image_path, "rb") as f:
        image_bytes = f.read()
    start_time = time()
    text, confidence = ocr_service.extract_text(image_bytes)
    print(f"OCR Confidence: {confidence:.2%}")
    print(f"\nExtracted Text:\n{text}")
    end_time = time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")



# uv run src/services/processors/image_processor.py