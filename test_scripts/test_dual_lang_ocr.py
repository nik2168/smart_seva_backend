"""
Test file for dual-language (Telugu + English) PaddleOCR
Demonstrates running both Telugu and English OCR instances and merging results

Usage:
    cd server
    uv run test_data/test_dual_lang_ocr.py

Or:
    python test_data/test_dual_lang_ocr.py

This test:
1. Initializes separate Telugu and English PaddleOCR instances
2. Runs both OCR engines on the same image
3. Merges results intelligently (deduplication + confidence-based)
4. Shows formatted output with statistics

Test images:
- telgu.jpeg (Telugu text)
- hand.png (English form with Telugu names)
"""
import os
import sys
import time
from pathlib import Path

# Add parent directory to path to import OCRService
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from paddleocr import PaddleOCR
    import paddle
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    print("PaddleOCR not available. Please install: pip install paddleocr")
    sys.exit(1)

# Import settings
from src.config import settings


# Script detection helpers
def is_telugu(text: str) -> bool:
    """Check if text contains Telugu script characters"""
    return any("\u0C00" <= ch <= "\u0C7F" for ch in text)


def is_english(text: str) -> bool:
    """Check if text contains English alphabetic characters"""
    return any(ch.isalpha() and ch.isascii() for ch in text)


class DualLanguageOCR:
    """Dual-language OCR using separate Telugu and English PaddleOCR instances"""
    
    def __init__(self):
        """Initialize both Telugu and English OCR instances"""
        self.ocr_te = None  # Telugu OCR instance
        self.ocr_en = None  # English OCR instance
        
        # Configure PaddlePaddle for memory optimization
        if paddle is not None:
            try:
                paddle.set_flags({
                    "FLAGS_fraction_of_cpu_memory_to_use": 0.85,
                    "FLAGS_use_pinned_memory": False,
                })
            except Exception:
                pass
        
        print("Initializing Telugu PaddleOCR...")
        try:
            self.ocr_te = PaddleOCR(
                use_textline_orientation=True,
                lang="te",
                cpu_threads=2,
                enable_mkldnn=False,
            )
            print("✓ Telugu OCR initialized")
        except Exception as e:
            print(f"✗ Telugu OCR initialization failed: {e}")
        
        print("Initializing English PaddleOCR...")
        try:
            self.ocr_en = PaddleOCR(
                use_textline_orientation=True,
                lang="en",
                cpu_threads=2,
                enable_mkldnn=False,
            )
            print("✓ English OCR initialized")
        except Exception as e:
            print(f"✗ English OCR initialization failed: {e}")
    
    def extract_text_dual(self, image_path: str) -> dict:
        """
        Extract text using both Telugu and English OCR, then merge results
        
        Behavior controlled by settings.ocr_bi_lang:
        - If True: Always run both OCRs and combine results (bilingual mode)
        - If False: Run English OCR first, then Telugu OCR only if needed (optimized mode)
        
        Returns:
            dict with keys:
                - 'telugu': {'texts': [...], 'scores': [...], 'full_text': str}
                - 'english': {'texts': [...], 'scores': [...], 'full_text': str}
                - 'merged': {'texts': [...], 'scores': [...], 'full_text': str}
                - 'stats': {'avg_confidence': float, 'total_blocks': int}
        """
        results = {
            'telugu': {'texts': [], 'scores': [], 'full_text': ''},
            'english': {'texts': [], 'scores': [], 'full_text': ''},
            'merged': {'texts': [], 'scores': [], 'full_text': ''},
        }
        
        # Check bilingual mode setting
        bi_lang_mode = getattr(settings, 'ocr_bi_lang', False)
        mode_str = "Bilingual Mode (always run both)" if bi_lang_mode else "Optimized Mode (English first, Telugu if needed)"
        print(f"\n[OCR Mode] {mode_str}")
        
        if bi_lang_mode:
            # BILINGUAL MODE: Always run both OCRs and combine results
            print("\n[Bilingual Mode] Enabled - Running both Telugu and English OCR...")
            
            # Run English OCR
            if self.ocr_en:
                print("\n[English OCR] Processing...")
                try:
                    result_en = self.ocr_en.predict(image_path)
                    if result_en and isinstance(result_en, list) and len(result_en) > 0:
                        block = result_en[0]
                        texts_en = block.get("rec_texts", [])
                        scores_en = block.get("rec_scores", [])
                        
                        for text, score in zip(texts_en, scores_en):
                            if text and isinstance(text, str):
                                text = text.strip()
                                confidence = float(score) if score else 1.0
                                is_meaningful = len(text) > 1 or text.isalnum()
                                if text and confidence >= 0.5 and is_meaningful:
                                    results['english']['texts'].append(text)
                                    results['english']['scores'].append(confidence)
                        
                        results['english']['full_text'] = "\n".join(results['english']['texts'])
                        print(f"✓ English OCR: {len(results['english']['texts'])} text blocks")
                except Exception as e:
                    print(f"✗ English OCR failed: {e}")
            
            # Always run Telugu OCR in bilingual mode
            if self.ocr_te:
                print("\n[Telugu OCR] Processing (bilingual mode)...")
                try:
                    result_te = self.ocr_te.predict(image_path)
                    if result_te and isinstance(result_te, list) and len(result_te) > 0:
                        block = result_te[0]
                        texts_te = block.get("rec_texts", [])
                        scores_te = block.get("rec_scores", [])
                        
                        for text, score in zip(texts_te, scores_te):
                            if text and isinstance(text, str):
                                text = text.strip()
                                confidence = float(score) if score else 1.0
                                
                                # Skip English lines detected by Telugu OCR (duplication prevention)
                                if not is_telugu(text) and is_english(text):
                                    continue
                                
                                if confidence < 0.4:
                                    continue
                                
                                if len(text) <= 1:
                                    continue
                                
                                results['telugu']['texts'].append(text)
                                results['telugu']['scores'].append(confidence)
                        
                        results['telugu']['full_text'] = "\n".join(results['telugu']['texts'])
                        print(f"✓ Telugu OCR: {len(results['telugu']['texts'])} text blocks")
                except Exception as e:
                    print(f"✗ Telugu OCR failed: {e}")
        
        else:
            # OPTIMIZED MODE: Run English OCR first, then decide if Telugu OCR is needed
            if self.ocr_en:
                print("\n[English OCR] Processing...")
                try:
                    result_en = self.ocr_en.predict(image_path)  # Use predict() method (newer API)
                    if result_en and isinstance(result_en, list) and len(result_en) > 0:
                        block = result_en[0]
                        texts_en = block.get("rec_texts", [])
                        scores_en = block.get("rec_scores", [])
                        
                        for text, score in zip(texts_en, scores_en):
                            if text and isinstance(text, str):
                                text = text.strip()
                                confidence = float(score) if score else 1.0
                                # Filter out low-confidence and garbage results
                                is_meaningful = len(text) > 1 or text.isalnum()
                                if text and confidence >= 0.5 and is_meaningful:
                                    results['english']['texts'].append(text)
                                    results['english']['scores'].append(confidence)
                        
                        results['english']['full_text'] = "\n".join(results['english']['texts'])
                        print(f"✓ English OCR: {len(results['english']['texts'])} text blocks")
                except Exception as e:
                    print(f"✗ English OCR failed: {e}")
            
            # Check if Telugu text is present in English OCR results
            has_telugu_in_english = any(is_telugu(text) for text in results['english']['texts'])
            
            # Check if English OCR might have failed on Telugu-only document
            # Indicators: very few results, low confidence, or mostly non-English text
            english_result_count = len(results['english']['texts'])
            avg_english_conf = sum(results['english']['scores']) / len(results['english']['scores']) if results['english']['scores'] else 0
            
            # Check if English OCR returned mostly non-English text (might be Telugu misread)
            english_text_ratio = sum(1 for t in results['english']['texts'] if is_english(t)) / max(english_result_count, 1)
            
            # Run Telugu OCR if:
            # 1. Telugu detected in English OCR (bilingual or Telugu doc)
            # 2. English OCR returned very few results (< 3) - might be Telugu-only
            # 3. English OCR has low confidence (< 0.6) - might be Telugu-only
            # 4. English OCR returned mostly non-English text (< 30% English) - likely Telugu-only
            should_run_telugu = (
                has_telugu_in_english or 
                english_result_count < 3 or 
                avg_english_conf < 0.6 or 
                english_text_ratio < 0.3
            )
            
            if should_run_telugu and self.ocr_te:
                reason = []
                if has_telugu_in_english:
                    reason.append("Telugu text detected")
                if english_result_count < 3:
                    reason.append(f"few results ({english_result_count})")
                if avg_english_conf < 0.6:
                    reason.append(f"low confidence ({avg_english_conf:.2f})")
                if english_text_ratio < 0.3:
                    reason.append(f"mostly non-English ({english_text_ratio:.1%})")
                
                print(f"\n[Telugu OCR] Processing ({', '.join(reason)})...")
                try:
                    result_te = self.ocr_te.predict(image_path)  # Use predict() method (newer API)
                    if result_te and isinstance(result_te, list) and len(result_te) > 0:
                        block = result_te[0]
                        texts_te = block.get("rec_texts", [])
                        scores_te = block.get("rec_scores", [])
                        
                        for text, score in zip(texts_te, scores_te):
                            if text and isinstance(text, str):
                                text = text.strip()
                                confidence = float(score) if score else 1.0
                                
                                # Skip English lines detected by Telugu OCR (duplication prevention)
                                if not is_telugu(text) and is_english(text):
                                    continue  # Telugu OCR accidentally read English → skip
                                
                                # Skip low-confidence results
                                if confidence < 0.4:
                                    continue
                                
                                # Skip single-character garbage
                                if len(text) <= 1:
                                    continue
                                
                                results['telugu']['texts'].append(text)
                                results['telugu']['scores'].append(confidence)
                        
                        results['telugu']['full_text'] = "\n".join(results['telugu']['texts'])
                        print(f"✓ Telugu OCR: {len(results['telugu']['texts'])} text blocks")
                except Exception as e:
                    print(f"✗ Telugu OCR failed: {e}")
            elif not should_run_telugu:
                print("\n[Telugu OCR] Skipping (English OCR sufficient - no Telugu detected)")
        
        # Merge results intelligently - PRIORITIZE ENGLISH in bilingual documents
        print("\n[Merging Results] Combining Telugu + English (English preferred)...")
        merged_texts = []
        merged_scores = []
        seen_texts = set()  # For deduplication
        
        # Detect if document is bilingual
        has_telugu = any(is_telugu(t) for t in results['telugu']['texts'])
        has_english = len(results['english']['texts']) > 0
        is_bilingual = has_telugu and has_english
        
        if is_bilingual:
            print("  → Bilingual document detected: English will be prioritized")
        
        # Process English results FIRST (for bilingual docs, English takes priority)
        for text, score in zip(results['english']['texts'], results['english']['scores']):
            # Skip very short or non-meaningful English text
            if len(text) <= 2 and not text.isalnum():
                continue
            
            # Skip low-confidence English results
            if score < 0.5:
                continue
            
            if text not in seen_texts:
                merged_texts.append(text)
                merged_scores.append(min(score, 1.0))  # Cap at 1.0 (100%)
                seen_texts.add(text)
        
        # Process Telugu results SECOND (add only if not duplicate or if Telugu-only)
        for text, score in zip(results['telugu']['texts'], results['telugu']['scores']):
            # Skip English lines detected by Telugu OCR (already handled by English OCR)
            if not is_telugu(text) and is_english(text):
                continue  # Skip - English OCR already handled this
            
            # Skip low-confidence Telugu results
            if score < 0.4:
                continue
            
            if text not in seen_texts:
                # Telugu-only text - add it
                merged_texts.append(text)
                merged_scores.append(min(score, 1.0))  # Cap at 1.0 (100%)
                seen_texts.add(text)
            else:
                # If duplicate exists (English already added it), prefer English version
                # Only replace if Telugu confidence is significantly higher (rare case)
                idx = merged_texts.index(text)
                if score > merged_scores[idx] * 1.15:  # Telugu must be 15% better to override
                    merged_scores[idx] = min(score, 1.0)  # Cap at 1.0
        
        # Final cleanup: Remove numeric-only or symbol-only lines
        cleaned_texts = []
        cleaned_scores = []
        for text, score in zip(merged_texts, merged_scores):
            # Keep lines that have at least one alphabetic character
            if any(ch.isalpha() for ch in text):
                cleaned_texts.append(text)
                cleaned_scores.append(score)
        
        results['merged']['texts'] = cleaned_texts
        results['merged']['scores'] = cleaned_scores
        results['merged']['full_text'] = "\n".join(cleaned_texts)
        
        # Calculate statistics
        all_scores = cleaned_scores if cleaned_scores else [0.0]
        # Cap scores at 100% for statistics
        capped_scores = [min(s, 1.0) for s in all_scores]
        results['stats'] = {
            'avg_confidence': sum(capped_scores) / len(capped_scores) * 100 if capped_scores else 0.0,
            'min_confidence': min(capped_scores) * 100 if capped_scores else 0.0,
            'max_confidence': min(max(capped_scores), 100.0) if capped_scores else 0.0,  # Cap at 100%
            'total_blocks': len(cleaned_texts),
            'telugu_blocks': len(results['telugu']['texts']),
            'english_blocks': len(results['english']['texts']),
            'merged_blocks': len(cleaned_texts),
            'is_bilingual': is_bilingual,  # Document type detection
        }
        
        return results
    
    def print_results(self, results: dict):
        """Print formatted results"""
        print("\n" + "=" * 80)
        print("DUAL-LANGUAGE OCR RESULTS")
        print("=" * 80)
        
        # Statistics
        stats = results['stats']
        print(f"\n📊 Statistics:")
        doc_type = "Bilingual (English prioritized)" if stats.get('is_bilingual', False) else "Single-language"
        print(f"   Document type: {doc_type}")
        print(f"   Telugu blocks: {stats['telugu_blocks']}")
        print(f"   English blocks: {stats['english_blocks']}")
        print(f"   Merged blocks: {stats.get('merged_blocks', stats['total_blocks'])}")
        print(f"   Average confidence: {stats['avg_confidence']:.2f}%")
        print(f"   Min confidence: {stats['min_confidence']:.2f}%")
        print(f"   Max confidence: {stats['max_confidence']:.2f}%")
        
        # Telugu results
        if results['telugu']['texts']:
            print("\n" + "-" * 80)
            print("📝 TELUGU OCR RESULTS:")
            print("-" * 80)
            for idx, (text, score) in enumerate(zip(results['telugu']['texts'], results['telugu']['scores']), 1):
                display_score = min(score * 100, 100.0)
                print(f"  {idx:2d}. [{display_score:5.1f}%] {text}")
        
        # English results
        if results['english']['texts']:
            print("\n" + "-" * 80)
            print("📝 ENGLISH OCR RESULTS:")
            print("-" * 80)
            for idx, (text, score) in enumerate(zip(results['english']['texts'], results['english']['scores']), 1):
                display_score = min(score * 100, 100.0)
                print(f"  {idx:2d}. [{display_score:5.1f}%] {text}")
        
        # Merged results
        print("\n" + "-" * 80)
        print("🔀 MERGED RESULTS (Telugu + English):")
        print("-" * 80)
        for idx, (text, score) in enumerate(zip(results['merged']['texts'], results['merged']['scores']), 1):
            # Cap display at 100%
            display_score = min(score * 100, 100.0)
            print(f"  {idx:2d}. [{display_score:5.1f}%] {text}")
        
        # Full merged text
        print("\n" + "=" * 80)
        print("📄 FULL EXTRACTED TEXT (Merged):")
        print("=" * 80)
        print(results['merged']['full_text'])
        print("=" * 80)


def main():
    """
    Main test function
    
    To enable bilingual mode (always run both OCRs):
    - Set OCR_BI_LANG=true in .env file, OR
    - Set settings.ocr_bi_lang = True in code
    
    Bilingual mode: Always runs both Telugu and English OCR and combines results
    Optimized mode: Runs English OCR first, then Telugu OCR only if needed
    """
    # Find test images
    test_dir = Path(__file__).parent
    test_images = [
        test_dir / "telgu.jpeg",    # Telugu image
    ]
    
    # Find available test image
    test_image = None
    for img_path in test_images:
        if img_path.exists():
            test_image = img_path
            break
    
    if not test_image:
        print("❌ No test image found. Please add telgu.jpeg or hand.png to test_data/")
        return
    
    print(f"📸 Using test image: {test_image.name}")
    print(f"📁 Full path: {test_image}")
    
    # Initialize dual-language OCR
    dual_ocr = DualLanguageOCR()
    
    if not dual_ocr.ocr_te and not dual_ocr.ocr_en:
        print("❌ Failed to initialize OCR instances")
        return
    
    # Run OCR
    print(f"\n🚀 Starting dual-language OCR on {test_image.name}...")
    start_time = time.time()
    
    results = dual_ocr.extract_text_dual(str(test_image))
    
    elapsed_time = time.time() - start_time
    
    # Print results
    dual_ocr.print_results(results)
    
    print(f"\n⏱️  Total time: {elapsed_time:.2f} seconds")
    print("\n✅ Test completed!")


if __name__ == "__main__":
    main()


# uv run test_data/test_dual_lang_ocr.py