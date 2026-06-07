from openai import OpenAI
from sqlalchemy.orm import Session
from sqlalchemy import text
from dotenv import load_dotenv
import os
import json

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

def get_embedding(text_input: str) -> list[float]:
    response = client.embeddings.create(
        model="openai/text-embedding-3-small",
        input=text_input
    )
    return response.data[0].embedding

def chunk_text(text: str, chunk_size: int = 500) -> list[str]:
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        current_chunk.append(word)
        current_length += len(word) + 1
        if current_length >= chunk_size:
            chunks.append(" ".join(current_chunk))
            overlap_words = current_chunk[-10:]
            current_chunk = overlap_words
            current_length = sum(len(w) + 1 for w in overlap_words)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def embed_document(document_id: int, content: str, db: Session):
    # delete old embeddings for this document
    db.execute(
        text("DELETE FROM document_embeddings WHERE document_id = :doc_id"),
        {"doc_id": document_id}
    )

    chunks = chunk_text(content)

    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        # store embedding as JSON string instead of vector type
        db.execute(text("""
            INSERT INTO document_embeddings
            (document_id, chunk_index, chunk_text, embedding)
            VALUES (:doc_id, :chunk_idx, :chunk_text, :embedding)
        """), {
            "doc_id": document_id,
            "chunk_idx": i,
            "chunk_text": chunk,
            "embedding": json.dumps(embedding)   # JSON instead of vector
        })

    db.commit()
    return len(chunks)