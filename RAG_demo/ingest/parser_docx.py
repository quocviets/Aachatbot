from docx import Document

def parse_docx(path):
    doc = Document(path)

    data = []
    current_crop = None
    current_disease = None

    for para in doc.paragraphs:
        text = para.text.strip()

        if not text:
            continue

        if text in ["Cây lúa", "Dưa leo", "Ngô", "Ớt", "Cà chua", "Táo"]:
            current_crop = text
            continue

        if text[0].isdigit():
            current_disease = text
            continue

        if current_crop and current_disease:
            data.append({
                "content": text,
                "metadata": {
                    "crop": current_crop,
                    "disease": current_disease
                }
            })

    return data