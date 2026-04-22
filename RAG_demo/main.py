import os
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import PGVector
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

app = FastAPI()

# Lấy cấu hình từ Environment Variables của Render
DB_URL = os.getenv("DATABASE_URL")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Khởi tạo các thành phần AI
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_db = PGVector(
    connection_string=DB_URL,
    collection_name="nong_nghiep_data",
    embedding_function=embeddings
)

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)

# Cấu hình Prompt chuyên gia
template = """Bạn là một chuyên gia tư vấn kỹ thuật nông nghiệp. 
Hãy trả lời dựa trên ngữ cảnh: {context}
Câu hỏi: {question}
Trả lời chi tiết:"""
QA_PROMPT = PromptTemplate.from_template(template)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
    chain_type_kwargs={"prompt": QA_PROMPT}
)

class Msg(BaseModel):
    message: str

@app.post("/ask")
async def ask(request: Msg):
    res = qa_chain.invoke({"query": request.message})
    return {"answer": res["result"]}

@app.get("/")
def health():
    return {"status": "online"}