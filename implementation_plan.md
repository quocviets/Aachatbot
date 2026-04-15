# Refactor: Multi-Model Plant Disease Inference Pipeline (v2)

Pipeline **2 giai đoạn**: Stage 1 (nhận diện cây) → Stage 2 (phân loại bệnh), được refactor thành hệ thống modular, production-ready với 5 cải tiến so với v1.

---

## Proposed Changes

### Core — Shared Foundation

#### [NEW] [config.py](file:///c:/Users/lequo/Downloads/Đồ Án/Core/config.py) ✅ Done
- `BASE_DIR`, `MODEL_DIR`, device auto-detect, `IMAGE_SIZE`
- `STAGE1_CLASSES`, `STAGE2_CLASS_MAP`
- **[NEW v2]** `CONFIDENCE_THRESHOLD: float = 0.60` — dưới ngưỡng này trả về `"Unknown"`

---

#### [NEW] [exceptions.py](file:///c:/Users/lequo/Downloads/Đồ Án/Core/exceptions.py)
Custom exception hierarchy:
```python
InferenceError(Exception)          # base
  ├── ModelNotFoundError            # .pth file không tồn tại
  ├── InvalidImageError             # ảnh không đọc được
  └── UnsupportedPlantError         # cây không có trong danh sách
```

---

#### [NEW] [model_base.py](file:///c:/Users/lequo/Downloads/Đồ Án/Core/model_base.py)
Abstract interface **Dependency Inversion**:
```python
class BasePlantModel(ABC):
    @abstractmethod
    def predict(self, tensor) -> tuple[int, float]: ...

    @abstractmethod
    def get_classes(self) -> list[str]: ...
```
→ Đổi CNN → ViT sau này chỉ cần tạo class mới, không sửa `predictor.py` hay `ModelManager`.

---

#### [NEW] [model_arch.py](file:///c:/Users/lequo/Downloads/Đồ Án/Core/model_arch.py) ✅ Done
- [build_mobilenetv3_small(num_classes)](file:///c:/Users/lequo/Downloads/%C4%90%E1%BB%93%20%C3%81n/model.py#5-18) — moved từ root [model.py](file:///c:/Users/lequo/Downloads/%C4%90%E1%BB%93%20%C3%81n/model.py)

---

#### [NEW] [utils.py](file:///c:/Users/lequo/Downloads/Đồ Án/Core/utils.py) ✅ Done
- `setup_logger(name)` — logger chuẩn
- `validate_image_path(path)` — kiểm tra file tồn tại
- **[NEW v2]** `measure_time` decorator — wrap function, inject `inference_time_ms` vào result

---

### Inference — Pipeline Modules

#### [NEW] [model_manager.py](file:///c:/Users/lequo/Downloads/Đồ Án/Inference/model_manager.py)

Class `ModelManager` — quản lý **tất cả** model (Stage 1 và Stage 2) đồng nhất:

| Method | Mô tả |
|---|---|
| `_scan_models()` | Auto-scan `Model/` → build registry `name → path` |
| `get_model(name)` | Lazy load + cache, trả về `BasePlantModel` |
| `list_plants()` | Danh sách cây hỗ trợ (Stage 2) |

**Không còn special case Stage 1** — `"stage1"` được đăng ký như mọi model khác.

---

#### [NEW] [plant_models.py](file:///c:/Users/lequo/Downloads/Đồ Án/Inference/plant_models.py)

Concrete implementations của `BasePlantModel`:

```python
class Stage1Model(BasePlantModel):    # nhận diện loại cây
class Stage2Model(BasePlantModel):    # phân loại bệnh (dùng STAGE2_CLASS_MAP)
```

Cả hai gọi `torch.no_grad()` nội bộ, trả về [(class_idx, confidence_float)](file:///c:/Users/lequo/Downloads/%C4%90%E1%BB%93%20%C3%81n/Inference.py#167-182).

---

#### [NEW] [preprocess.py](file:///c:/Users/lequo/Downloads/Đồ Án/Inference/preprocess.py)
Class `ImagePreprocessor`:
- `load_and_transform(image_path)` → tensor `[1, 3, H, W]`
- Raise `InvalidImageError` nếu ảnh lỗi

---

#### [NEW] [postprocess.py](file:///c:/Users/lequo/Downloads/Đồ Án/Inference/postprocess.py)
`format_result(...)` → dict chuẩn:
```python
{
    "plant": "Apple",
    "disease": "Apple_scab",       # hoặc "Unknown" nếu dưới threshold
    "confidence": 0.94,
    "plant_confidence": 0.98,
    "inference_time_ms": 120.5,
    "supported": True
}
```

---

#### [NEW] [predictor.py](file:///c:/Users/lequo/Downloads/Đồ Án/Inference/predictor.py)

Entry point `PlantDiseasePredictor`:
- [predict(image_path)](file:///c:/Users/lequo/Downloads/%C4%90%E1%BB%93%20%C3%81n/Inference.py#167-182) — decorated bằng `@measure_time`
- Dùng `ModelManager.get_model("stage1")` → không còn load Stage 1 riêng
- Kiểm tra confidence threshold sau mỗi stage

---

### Folder Structure Cuối

```
Đồ Án/
├── Core/
│   ├── __init__.py
│   ├── config.py          ✅
│   ├── exceptions.py      ← NEW v2
│   ├── model_base.py      ← NEW v2
│   ├── model_arch.py      ✅
│   └── utils.py           ✅
│
├── Inference/
│   ├── __init__.py
│   ├── plant_models.py    ← NEW v2 (Stage1Model, Stage2Model)
│   ├── model_manager.py
│   ├── preprocess.py
│   ├── postprocess.py
│   └── predictor.py
│
├── Model/
│   ├── Stage 1/           ← quản lý bởi ModelManager như mọi model
│   ├── Apple/
│   ├── Corn/
│   └── ...
│
├── Train/  Dataset/  RAG/  Test/   (không đổi)
└── model.py   (deprecated, giữ lại để backward compat)
```

---

## Verification Plan

```bash
python -c "
from Inference.predictor import PlantDiseasePredictor
p = PlantDiseasePredictor()

# Test 1: list plants
print(p.list_supported_plants())

# Test 2: predict ảnh có sẵn
result = p.predict(r'Inference/ngô đốm lá.jpg')
print(result)

# Test 3: predict 2 lần → model không reload (check log)
result2 = p.predict(r'Inference/ngô đốm lá.jpg')
print(result2)
"
```
