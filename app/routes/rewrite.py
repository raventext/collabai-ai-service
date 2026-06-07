from fastapi import APIRouter
from pydantic import BaseModel
from typing import Literal
from openai import OpenAI
import os

router = APIRouter()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"),base_url="https://openrouter.ai/api/v1")

class RewriteRequest(BaseModel):
    text: str
    mode: Literal["improve", "formal", "casual", "shorter", "longer"]

class RewriteResponse(BaseModel):
    rewritten: str

MODE_PROMPTS = {
    "improve": "Improve the clarity and quality of this text while keeping the same meaning:",
    "formal": "Rewrite this text in a formal, professional tone:",
    "casual": "Rewrite this text in a casual, friendly tone:",
    "shorter": "Make this text shorter and more concise without losing key information:",
    "longer": "Expand this text with more detail and explanation:"
}

@router.post("/rewrite", response_model=RewriteResponse)
async def rewrite(request: RewriteRequest):
    prompt = MODE_PROMPTS.get(request.mode, MODE_PROMPTS["improve"])

    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": f"{prompt}\n\n{request.text}\n\nReturn only the rewritten text, nothing else."
            }
        ]
    )
    return RewriteResponse(rewritten=response.choices[0].message.content)