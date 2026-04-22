from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_data(data):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    chunks = []

    for item in data:
        texts = splitter.split_text(item["content"])
        for t in texts:
            chunks.append({
                "content": t,
                "metadata": item["metadata"]
            })

    return chunks