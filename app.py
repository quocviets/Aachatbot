"""
Simple Plant Disease API
========================
Run: uvicorn app:app --reload  (từ thư mục gốc Đồ Án/)

Endpoints:
  POST /auth/login   → lấy token
  POST /predict      → upload ảnh, nhận kết quả
  GET  /health       → kiểm tra server
"""

import os
import sys
import uuid
import tempfile

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

# ── Thêm project root vào path để import Inference ──────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from Inference.pipeline import predict_image  # noqa: E402

# ════════════════════════════════════════════════════════════════════════════
# CONFIG — đổi ở đây nếu cần
# ════════════════════════════════════════════════════════════════════════════

DEMO_USERNAME = "admin"
DEMO_PASSWORD = "123456"
DEMO_TOKEN    = "demo-token-123"

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# ════════════════════════════════════════════════════════════════════════════
# APP
# ════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="🌿 Plant Disease API",
    description="AI-powered plant disease classification — demo version",
    version="1.0.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # mobile app có thể gọi từ mọi nơi
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

# ════════════════════════════════════════════════════════════════════════════
# SCHEMAS
# ════════════════════════════════════════════════════════════════════════════

class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "Bearer"

class PredictResponse(BaseModel):
    plant: str
    disease: str | None
    confidence: float
    inference_time_ms: float

# ════════════════════════════════════════════════════════════════════════════
# AUTH HELPER
# ════════════════════════════════════════════════════════════════════════════

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Dependency: xác minh Bearer token. Raise 401 nếu sai."""
    if credentials.credentials != DEMO_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

# ════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ════════════════════════════════════════════════════════════════════════════

@app.get("/health", tags=["System"])
def health():
    """Kiểm tra server còn sống không."""
    return {"status": "ok"}


@app.post("/auth/login", response_model=LoginResponse, tags=["Auth"])
def login(body: LoginRequest):
    """
    Đăng nhập bằng username/password hardcode.
    Trả về token để dùng cho /predict.
    """
    if body.username != DEMO_USERNAME or body.password != DEMO_PASSWORD:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Wrong username or password",
        )
    return LoginResponse(access_token=DEMO_TOKEN)


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(
    file: UploadFile = File(..., description="Ảnh lá cây (JPG/PNG)"),
    token: str = Depends(verify_token),
):
    """
    Upload ảnh → nhận kết quả phân loại bệnh cây.

    **Yêu cầu header:**
    ```
    Authorization: Bearer demo-token-123
    ```
    """
    # ── 1. Validate định dạng file ──────────────────────────────────────────
    filename = file.filename or "upload.jpg"
    ext = os.path.splitext(filename)[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type '{ext}' not supported. Use JPG or PNG.",
        )

    # ── 2. Đọc bytes ────────────────────────────────────────────────────────
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="File is empty")

    # ── 3. Lưu tạm rồi gọi inference ────────────────────────────────────────
    suffix = ext if ext else ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        result: dict = predict_image(tmp_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc
    finally:
        os.unlink(tmp_path)   # xóa file tạm

    # ── 4. Trả kết quả ──────────────────────────────────────────────────────
    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("message", "Unknown error"))

    return PredictResponse(
        plant=result.get("plant", "Unknown"),
        disease=result.get("disease"),
        confidence=result.get("disease_confidence") or result.get("plant_confidence") or 0.0,
        inference_time_ms=result.get("inference_time_ms", 0.0),
    )
