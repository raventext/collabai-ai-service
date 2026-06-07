from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from app.database import get_db
from app.rag.embedder import embed_document

router = APIRouter()

class EmbedRequest(BaseModel):
    document_id: int
    content: str

@router.post("/embed")
async def embed(request: EmbedRequest, db: Session = Depends(get_db)):
    count = embed_document(request.document_id, request.content, db)
    return {"message": f"Embedded {count} chunks for document {request.document_id}"}