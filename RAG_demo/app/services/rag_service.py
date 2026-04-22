from app.services.retriever import search
from app.services.llm_service import generate_answer

def rag_pipeline(query):
    docs = search(query)

    context = "\n".join(docs)

    answer = generate_answer(query, context)

    return answer