from paddleocr import PaddleOCR
from time import time

# Initialize PaddleOCR with updated API (use_textline_orientation instead of use_angle_cls)
ocr = PaddleOCR(use_textline_orientation=True, lang="en")

# Call ocr() without deprecated det/cls parameters
start_time = time()
result = ocr.ocr("test_data/hand2.png")
end_time = time()
print(f"Time taken: {end_time - start_time:.2f} seconds")
text = result[0].get("rec_texts", [])

for t in text:
    print(t)




# uv run test_data/pad_test.py