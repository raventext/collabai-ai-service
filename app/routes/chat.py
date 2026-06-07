from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from openai import OpenAI
import os

from app.database import get_db
from app.rag.retriever import retrieve_relevant_chunks

router = APIRouter()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"),base_url="https://openrouter.ai/api/v1")

class ChatRequest(BaseModel):
    document_id: int
    question: str

class ChatResponse(BaseModel):
    answer: str
    sources: list[str]

@router.post("/chat", response_model=ChatResponse)
async def chat_with_document(
        request: ChatRequest,
        db: Session = Depends(get_db)):

    relevant_chunks = retrieve_relevant_chunks(
        document_id=request.document_id,
        question=request.question,
        db=db
    )

    if not relevant_chunks:
        return ChatResponse(
            answer="I couldn't find relevant information in this document to answer your question.",
            sources=[]
        )

    context = "\n\n---\n\n".join(relevant_chunks)

    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": f"""You are a helpful assistant. Answer the question based ONLY on the provided document context.
If the answer is not in the context, say "I don't have enough information in this document to answer that."

Document Context:
{context}

Question: {request.question}

Answer:"""
            }
        ]
    )

    return ChatResponse(
        answer=response.choices[0].message.content,
        sources=relevant_chunks
    )