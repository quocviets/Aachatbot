# Agri-AI RAG API
Hệ thống tư vấn bệnh cây trồng sử dụng RAG, PostgreSQL và Gemini.

## Cách chạy Local:
1. Cài đặt thư viện: `pip install -r requirements.txt`
2. Chạy server: `uvicorn main:app --reload`
3. Truy cập Swagger UI: `http://localhost:8000/docs`

## Kết nối Mobile App:
Gửi POST request tới `/ask` với body: `{"message": "câu hỏi của bạn"}`