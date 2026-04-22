from fastapi import APIRouter
from app.services.rag_service import rag_pipeline

router = APIRouter()

@router.get("/ask")
def ask(query: str):
    answer = rag_pipeline(query)
    return {"question": query, "answer": answer}