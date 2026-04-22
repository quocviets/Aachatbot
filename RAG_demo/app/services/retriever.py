from sentence_transformers import SentenceTransformer
from app.core.database import get_db

model = SentenceTransformer("all-MiniLM-L6-v2")

def search(query, crop=None, top_k=5):
    conn = get_db()
    cur = conn.cursor()

    q_emb = model.encode(query).tolist()

    if crop:
        cur.execute("""
            SELECT content
            FROM documents
            WHERE metadata->>'crop' = %s
            ORDER BY embedding <-> %s
            LIMIT %s
        """, (crop, q_emb, top_k))
    else:
        cur.execute("""
            SELECT content
            FROM documents
            ORDER BY embedding <-> %s
            LIMIT %s
        """, (q_emb, top_k))

    results = [r[0] for r in cur.fetchall()]

    cur.close()
    conn.close()

    return results