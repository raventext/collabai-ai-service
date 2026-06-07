from sqlalchemy.orm import Session
from sqlalchemy import text
from app.rag.embedder import get_embedding
import json
import math

def cosine_similarity(vec1: list, vec2: list) -> float:
    """Calculate cosine similarity between two vectors in pure Python"""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))
    if magnitude1 == 0 or magnitude2 == 0:
        return 0
    return dot_product / (magnitude1 * magnitude2)

def retrieve_relevant_chunks(
        document_id: int,
        question: str,
        db: Session,
        top_k: int = 4) -> list[str]:

    # get question embedding
    question_embedding = get_embedding(question)

    # fetch all chunks for this document
    results = db.execute(text("""
        SELECT chunk_text, embedding
        FROM document_embeddings
        WHERE document_id = :doc_id
    """), {"doc_id": document_id})

    rows = results.fetchall()

    if not rows:
        return []

    # calculate similarity for each chunk
    scored_chunks = []
    for row in rows:
        chunk_text = row[0]
        chunk_embedding = json.loads(row[1])
        similarity = cosine_similarity(question_embedding, chunk_embedding)
        scored_chunks.append((similarity, chunk_text))

    # sort by similarity descending and return top_k
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored_chunks[:top_k]]