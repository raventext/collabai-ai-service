from fastapi import APIRouter
from pydantic import BaseModel
from openai import OpenAI
import os

router = APIRouter()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"),base_url="https://openrouter.ai/api/v1")

class SummarizeRequest(BaseModel):
    content: str
    document_id: int

class SummarizeResponse(BaseModel):
    summary: str

@router.post("/summarize", response_model=SummarizeResponse)
async def summarize(request: SummarizeRequest):
    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": f"""Please provide a clear, concise summary of the following document.

Document:
{request.content}

Provide a summary in 3-5 sentences covering the main points."""
            }
        ]
    )
    return SummarizeResponse(summary=response.choices[0].message.content)