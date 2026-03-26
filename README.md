# Plant Disease AI API

Backend API cho ứng dụng nhận diện bệnh cây trồng sử dụng AI.

## 🚀 Quick Start

### 1. Prerequisites
- Docker & Docker Compose
- Model files trong thư mục `Model/`

### 2. Setup
```bash
# Clone/copy project
cd /path/to/project

# Copy env template
cp .env.example .env

# Edit .env nếu cần (optional)
nano .env
```

### 3. Run with Docker
```bash
# Build và run
docker-compose up --build

# Hoặc background
docker-compose up -d --build
```

### 4. Test API
```bash
# Health check
curl http://localhost:8000/api/v1/health

# Get supported plants
curl http://localhost:8000/api/v1/plants

# Model info
curl http://localhost:8000/api/v1/model/info
```

## 📡 API Endpoints

### POST /api/v1/predict
Upload ảnh và nhận kết quả dự đoán.

**Request:**
- `file`: JPG/PNG image (max 5MB)
- `plant_type`: Optional (bỏ qua Stage 1)

**Response:**
```json
{
  "id": "uuid",
  "plant": "Apple",
  "plant_confidence": 0.97,
  "disease": "Apple_scab",
  "disease_confidence": 0.94,
  "inference_time_ms": 120.5,
  "image_url": "/uploads/uuid.jpg",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### GET /api/v1/history
Lịch sử dự đoán với pagination.

**Query params:**
- `page`: Page number (default: 1)
- `limit`: Items per page (max 50, default: 10)
- `date_from/date_to`: Date range (YYYY-MM-DD)
- `plant`: Filter by plant

### GET /api/v1/health
Health check + model status.

### GET /api/v1/plants
Danh sách cây hỗ trợ.

### GET /api/v1/model/info
Thông tin model + cache status.

## 🔧 Configuration

Sửa file `.env`:

```bash
# Database (SQLite default)
DATABASE_URL=sqlite+aiosqlite:///./plant_disease.db

# CORS origins
ALLOWED_ORIGINS=https://your-mobile-app.com,https://another-domain.com

# Rate limiting
RATE_LIMIT=100/minute

# Upload settings
MAX_FILE_SIZE_MB=10
```

## 🌐 Deploy Public

### Option 1: VPS/Cloud Server
```bash
# Trên server
git clone <your-repo>
cd project
cp .env.example .env
docker-compose up -d --build

# Setup nginx reverse proxy (optional)
# Expose port 8000 hoặc dùng domain
```

### Option 2: Railway/Replit/Render
- Push code lên GitHub
- Connect với platform free tier
- Set environment variables
- Deploy

### Option 3: Local + Ngrok (temporary)
```bash
# Install ngrok
npm install -g ngrok

# Run app locally
docker-compose up

# Expose public
ngrok http 8000
```

## 📱 Mobile Integration

### Example cURL:
```bash
curl -X POST "http://your-server:8000/api/v1/predict" \
  -F "file=@apple_leaf.jpg" \
  -F "plant_type=Apple"
```

### JavaScript (React Native):
```javascript
const formData = new FormData();
formData.append('file', {
  uri: imageUri,
  type: 'image/jpeg',
  name: 'leaf.jpg'
});

const response = await fetch('http://your-server:8000/api/v1/predict', {
  method: 'POST',
  body: formData
});

const result = await response.json();
```

## 🛠 Troubleshooting

### Model not loading
- Check `Model/` folder exists with `.pth` files
- Verify model architecture matches `build_mobilenetv3_small`

### Database issues
- Check write permissions for `plant_disease.db`
- Use absolute path if needed

### Port conflicts
- Change `APP_PORT` in `.env`
- Update docker-compose ports

## 📋 Requirements

- Python 3.11+
- PyTorch with CUDA (if GPU)
- FastAPI, SQLAlchemy, etc. (see `backend/requirements.txt`)