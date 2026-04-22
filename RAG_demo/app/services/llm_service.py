from openai import OpenAI
from app.core.config import settings

client = OpenAI(api_key=settings.OPENAI_API_KEY)

def generate_answer(query, context):
    prompt = f"""
Bạn là chuyên gia nông nghiệp.

Context:
{context}

Câu hỏi:
{query}

Trả lời chính xác:
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return response.choices[0].message.content