"""
Plant Disease API — Simple Version
===================================
Chạy: uvicorn main:app --reload
Docs: http://localhost:8000/docs
"""

import os
import sys
import tempfile

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

# ── Import model từ Inference/ ───────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from Inference.pipeline import predict_image as _ai_predict  # noqa: E402

# ════════════════════════════════
# CONFIG
# ════════════════════════════════
USERNAME    = "admin"
PASSWORD    = "123456"
VALID_TOKEN = "demo-token-123"

# ════════════════════════════════
# APP
# ════════════════════════════════
app = FastAPI(
    title="Plant Disease API",
    version="1.0.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

bearer = HTTPBearer()

# ════════════════════════════════
# SCHEMAS
# ════════════════════════════════
class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "Bearer"

class PredictResponse(BaseModel):
    label: str
    confidence: float

# ════════════════════════════════
# AUTH DEPENDENCY
# ════════════════════════════════
def require_token(cred: HTTPAuthorizationCredentials = Depends(bearer)):
    if cred.credentials != VALID_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")

# ════════════════════════════════
# ENDPOINTS
# ════════════════════════════════

@app.get("/health", tags=["System"])
def health():
    return {"status": "ok"}


@app.post("/auth/login", response_model=LoginResponse, tags=["Auth"])
def login(body: LoginRequest):
    if body.username != USERNAME or body.password != PASSWORD:
        raise HTTPException(status_code=401, detail="Wrong username or password")
    return LoginResponse(access_token=VALID_TOKEN)


@app.post("/predict", response_model=PredictResponse, tags=["Predict"])
async def predict(
    file: UploadFile = File(...),
    _: None = Depends(require_token),
):
    # Validate file
    ext = os.path.splitext(file.filename or "")[-1].lower()
    if ext not in {".jpg", ".jpeg", ".png"}:
        raise HTTPException(status_code=400, detail="Only JPG/PNG supported")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="File is empty")

    # Lưu tạm → gọi Inference
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext or ".jpg") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        result = _ai_predict(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)

    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("message"))

    # label = disease nếu có, fallback về plant
    label      = result.get("disease") or result.get("plant", "Unknown")
    confidence = result.get("disease_confidence") or result.get("plant_confidence") or 0.0

    return PredictResponse(label=label, confidence=round(confidence, 4))
