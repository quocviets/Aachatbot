import uuid
import json
from app.core.database import get_db
from ingest.parser_docx import parse_docx
from ingest.chunker import chunk_data
from ingest.embedder import embed

data = parse_docx("data/raw/Benh_cay.docx")
chunks = chunk_data(data)

conn = get_db()
cur = conn.cursor()

for c in chunks:
    emb = embed(c["content"])

    cur.execute("""
        INSERT INTO documents (id, content, metadata, embedding)
        VALUES (%s, %s, %s, %s)
    """, (
        str(uuid.uuid4()),
        c["content"],
        json.dumps(c["metadata"]),
        emb
    ))

conn.commit()
cur.close()
conn.close()

print("✅ DONE INGEST")