"""
Quick test script — chạy từ thư mục gốc project:
    python run_test.py
"""

import json
import os
import sys

# Đảm bảo import đúng từ root project
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Inference.pipeline import predict_image

# ── Đường dẫn ảnh test ──────────────────────────────────────────────────────
# Thay bằng ảnh bất kỳ của bạn
IMAGE_PATH = r"Inference\ngô đốm lá.jpg"

print("=" * 60)
print("TEST 1: Full pipeline (tự nhận diện cây qua Stage 1)")
print("=" * 60)
result = predict_image(IMAGE_PATH)
print(json.dumps(result, indent=2, ensure_ascii=False))

print()
print("=" * 60)
print("TEST 2: Bỏ qua Stage 1 — truyền plant_type='Corn' trực tiếp")
print("=" * 60)
result2 = predict_image(IMAGE_PATH, plant_type="Corn")
print(json.dumps(result2, indent=2, ensure_ascii=False))

print()
print("=" * 60)
print("TEST 3: Ảnh không tồn tại → trả về error dict")
print("=" * 60)
result3 = predict_image("không_tồn_tại.jpg")
print(json.dumps(result3, indent=2, ensure_ascii=False))

print()
print("=" * 60)
print("INFO: Các model đang available trong hệ thống")
print("=" * 60)
from Inference.model_manager import ModelManager
manager = ModelManager()
print("Available plants:", manager.available_plants())
