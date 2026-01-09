"""
Test script for Google Cloud Vision API OCR
Demonstrates how to use Google Vision API for text extraction

Usage:
    cd server
    uv run test_data/test_google_vision_ocr.py

Or:
    python test_data/test_google_vision_ocr.py

Prerequisites:
1. Install Google Cloud Vision library:
   pip install google-cloud-vision

2. Set up Google Cloud credentials:
   Option A: Set GOOGLE_VISION_API_KEY in .env file (JSON string or file path)
   Option B: Set GOOGLE_APPLICATION_CREDENTIALS environment variable to path of JSON key file
   Option C: Use gcloud auth application-default login

3. Enable Vision API in your Google Cloud project:
   - Go to https://console.cloud.google.com/apis/library/vision.googleapis.com
   - Click "Enable"

Test images:
- telgu.jpeg (Telugu text)
- bi_lang.png (Bilingual document)
- hand.png (English form with Telugu names)
"""
import os
import sys
import time
from pathlib import Path

# Add parent directory to path to import settings
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from google.cloud import vision
    from google.oauth2 import service_account
    import json
    GOOGLE_VISION_AVAILABLE = True
except ImportError:
    GOOGLE_VISION_AVAILABLE = False
    print("❌ Google Cloud Vision API not available.")
    print("   Please install: pip install google-cloud-vision")
    sys.exit(1)

# Import settings
from src.config import settings

# Temporarily set ocr_provider to google_vision for testing
# This allows testing Google Vision API directly


def get_vision_client():
    """Initialize and return Google Vision API client"""
    if not settings.google_vision_api_key:
        # Try environment variable as fallback
        api_key = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if api_key and os.path.isfile(api_key):
            credentials = service_account.Credentials.from_service_account_file(api_key)
            return vision.ImageAnnotatorClient(credentials=credentials)
        
        # Try default credentials
        try:
            return vision.ImageAnnotatorClient()
        except Exception as e:
            raise RuntimeError(
                "Google Vision API key not configured. Set GOOGLE_VISION_API_KEY in .env file "
                "or GOOGLE_APPLICATION_CREDENTIALS environment variable"
            )
    
    api_key = settings.google_vision_api_key.strip()
    
    # Remove any duplicate variable name if present (e.g., "GOOGLE_VISION_API_KEY=/path")
    if api_key.startswith("GOOGLE_VISION_API_KEY="):
        api_key = api_key.split("=", 1)[1].strip()
    
    # Resolve relative paths relative to server directory
    server_dir = Path(__file__).parent.parent
    if not os.path.isabs(api_key):
        # Try relative to server directory
        resolved_path = server_dir / api_key.lstrip('./')
        if resolved_path.exists():
            api_key = str(resolved_path)
    
    # Check if it's a JSON string or file path
    if api_key.startswith('{'):
        # JSON string provided
        try:
            credentials_info = json.loads(api_key)
            credentials = service_account.Credentials.from_service_account_info(credentials_info)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON in GOOGLE_VISION_API_KEY")
    elif os.path.isfile(api_key):
        # File path provided
        credentials = service_account.Credentials.from_service_account_file(api_key)
    else:
        # Try as JSON string
        try:
            credentials_info = json.loads(api_key)
            credentials = service_account.Credentials.from_service_account_info(credentials_info)
        except json.JSONDecodeError:
            raise ValueError(
                "GOOGLE_VISION_API_KEY must be a JSON file path or JSON string. "
                f"Got: {api_key[:50]}..."
            )
    
    return vision.ImageAnnotatorClient(credentials=credentials)


def extract_text_google_vision(image_path: str) -> dict:
    """
    Extract text from image using Google Cloud Vision API
    
    Returns:
        dict with keys:
            - 'text': Full extracted text
            - 'blocks': List of text blocks with bounding boxes
            - 'confidence': Average confidence (estimated)
            - 'total_blocks': Number of text blocks detected
    """
    client = get_vision_client()
    
    # Read image file
    with open(image_path, 'rb') as image_file:
        content = image_file.read()
    
    # Prepare image for Vision API
    image = vision.Image(content=content)
    
    # Perform text detection
    print(f"📸 Processing image: {Path(image_path).name}")
    print("🔍 Running Google Vision API text detection...")
    
    start_time = time.time()
    response = client.text_detection(image=image)
    elapsed_time = time.time() - start_time
    
    texts = response.text_annotations
    
    if not texts:
        print("⚠️  No text detected in image")
        return {
            'text': '',
            'blocks': [],
            'confidence': 0.0,
            'total_blocks': 0
        }
    
    # First element contains the full text
    full_text = texts[0].description if texts else ""
    
    # Extract individual text blocks with bounding boxes
    blocks = []
    for text in texts[1:]:  # Skip first element (full text)
        if text.description and text.description.strip():
            block_info = {
                'text': text.description.strip(),
                'confidence': 0.95,  # Google Vision doesn't provide per-block confidence
                'bounding_box': []
            }
            
            # Extract bounding box coordinates
            if text.bounding_poly and text.bounding_poly.vertices:
                for vertex in text.bounding_poly.vertices:
                    block_info['bounding_box'].append({
                        'x': vertex.x if hasattr(vertex, 'x') else 0,
                        'y': vertex.y if hasattr(vertex, 'y') else 0
                    })
            
            blocks.append(block_info)
    
    # Calculate statistics
    total_blocks = len(blocks)
    avg_confidence = 0.95  # Google Vision doesn't provide confidence scores
    
    print(f"✅ Text extraction completed in {elapsed_time:.2f} seconds")
    print(f"   Detected {total_blocks} text blocks")
    
    return {
        'text': full_text,
        'blocks': blocks,
        'confidence': avg_confidence,
        'total_blocks': total_blocks,
        'processing_time': elapsed_time
    }


def print_results(results: dict, image_name: str):
    """Print formatted results"""
    print("\n" + "=" * 80)
    print(f"GOOGLE VISION API OCR RESULTS - {image_name}")
    print("=" * 80)
    
    print(f"\n📊 Statistics:")
    print(f"   Total text blocks: {results['total_blocks']}")
    print(f"   Estimated confidence: {results['confidence']:.2%}")
    print(f"   Processing time: {results.get('processing_time', 0):.2f} seconds")
    
    # Individual blocks
    if results['blocks']:
        print("\n" + "-" * 80)
        print("📝 DETECTED TEXT BLOCKS:")
        print("-" * 80)
        for idx, block in enumerate(results['blocks'][:20], 1):  # Show first 20 blocks
            print(f"  {idx:2d}. [{block['confidence']:.0%}] {block['text']}")
            if block['bounding_box']:
                coords = block['bounding_box']
                print(f"      Bounding box: {coords}")
        
        if len(results['blocks']) > 20:
            print(f"  ... and {len(results['blocks']) - 20} more blocks")
    
    # Full text
    print("\n" + "=" * 80)
    print("📄 FULL EXTRACTED TEXT:")
    print("=" * 80)
    print(results['text'])
    print("=" * 80)


def main():
    """
    Main test function
    """
    print("🚀 Google Cloud Vision API OCR Test")
    print("=" * 80)
    
    # Check if Google Vision is available
    if not GOOGLE_VISION_AVAILABLE:
        print("❌ Google Cloud Vision API not available")
        print("   Install with: pip install google-cloud-vision")
        return
    
    # Find test images
    test_dir = Path(__file__).parent
    test_images = [
        # test_dir / "telgu.jpeg",    # Telugu image
        test_dir / "hand2.png",    # Bilingual image
        # test_dir / "hand.png",       # English form
    ]
    
    # Find available test image
    test_image = None
    for img_path in test_images:
        if img_path.exists():
            test_image = img_path
            break
    
    if not test_image:
        print("❌ No test image found.")
        print("   Please add one of these to test_data/: telgu.jpeg, bi_lang.png, or hand.png")
        return
    
    print(f"📸 Using test image: {test_image.name}")
    print(f"📁 Full path: {test_image}")
    
    # Check API key configuration
    if not settings.google_vision_api_key and not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        print("\n⚠️  Warning: Google Vision API key not configured")
        print("   Set GOOGLE_VISION_API_KEY in .env file or GOOGLE_APPLICATION_CREDENTIALS environment variable")
        print("\n   Options:")
        print("   1. Set GOOGLE_VISION_API_KEY in .env to JSON key file path")
        print("   2. Set GOOGLE_VISION_API_KEY in .env to JSON key content (as string)")
        print("   3. Set GOOGLE_APPLICATION_CREDENTIALS environment variable to key file path")
        print("   4. Use: gcloud auth application-default login")
        print("\n   Getting credentials:")
        print("   1. Go to https://console.cloud.google.com/apis/credentials")
        print("   2. Create credentials → Service Account Key")
        print("   3. Download JSON key file")
        print("   4. Add to .env: GOOGLE_VISION_API_KEY=/path/to/key.json")
        return
    
    try:
        # Run OCR
        print(f"\n🚀 Starting Google Vision API OCR on {test_image.name}...")
        start_time = time.time()
        
        results = extract_text_google_vision(str(test_image))
        
        total_time = time.time() - start_time
        
        # Print results
        print_results(results, test_image.name)
        
        print(f"\n⏱️  Total time: {total_time:.2f} seconds")
        print("\n✅ Test completed!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 Troubleshooting:")
        print("   1. Verify GOOGLE_VISION_API_KEY is set correctly in .env")
        print("   2. Ensure Vision API is enabled in your Google Cloud project")
        print("   3. Check that the service account has Vision API permissions")
        print("   4. Verify the JSON key file is valid")


if __name__ == "__main__":
    main()


# uv run test_data/test_google_vision_ocr.py