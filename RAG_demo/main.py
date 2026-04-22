import os

from fastapi import FastAPI, HTTPException
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import PGVector
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from pydantic import BaseModel


app = FastAPI()
qa_chain = None


class Msg(BaseModel):
    message: str


def get_qa_chain():
    global qa_chain

    if qa_chain is not None:
        return qa_chain

    db_url = os.getenv("DATABASE_URL")
    google_api_key = os.getenv("GOOGLE_API_KEY")

    if not db_url:
        raise RuntimeError("Missing DATABASE_URL environment variable")
    if not google_api_key:
        raise RuntimeError("Missing GOOGLE_API_KEY environment variable")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_db = PGVector(
        connection_string=db_url,
        collection_name="nong_nghiep_data",
        embedding_function=embeddings,
    )
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)

    prompt = PromptTemplate.from_template(
        """Ban la mot chuyen gia tu van ky thuat nong nghiep.
Hay tra loi dua tren ngu canh: {context}
Cau hoi: {question}
Tra loi chi tiet:"""
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": prompt},
    )
    return qa_chain


@app.get("/")
def health():
    return {"status": "online"}


@app.post("/ask")
async def ask(request: Msg):
    try:
        chain = get_qa_chain()
        result = chain.invoke({"query": request.message})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"answer": result["result"]}
