import os
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import PGVector

# 1. Cấu hình
os.environ["GOOGLE_API_KEY"] = "AIzaSy..." # API Key của bạn
CONNECTION_STRING = "postgresql+psycopg2://postgres.xxxx:Phamtheanh2901@aws-0-ap-southeast-1.pooler.supabase.com:6543/postgres"

# 2. Xử lý tài liệu
loader = Docx2txtLoader("Bệnh cây và cách chữa.docx")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = text_splitter.split_documents(documents)

# 3. Nạp vào Supabase (Sẽ xóa bảng cũ để làm mới hoàn toàn)
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
vector_db = PGVector.from_documents(
    embedding=embeddings,
    documents=chunks,
    collection_name="nong_nghiep_data",
    connection_string=CONNECTION_STRING,
    pre_delete_collection=True 
)
print("✅ Đã làm mới kho tri thức trên Supabase!")