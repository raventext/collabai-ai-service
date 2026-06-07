from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from app.database import SessionLocal, init_embeddings_table

load_dotenv()

from app.routes import summarize, rewrite, chat, embed

app = FastAPI(title="CollabAI - AI Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)

app.include_router(summarize.router, prefix="/api/ai", tags=["AI"])
app.include_router(rewrite.router, prefix="/api/ai", tags=["AI"])
app.include_router(chat.router, prefix="/api/ai", tags=["AI"])
app.include_router(embed.router, prefix="/api/ai", tags=["AI"])

@app.on_event("startup")
async def startup():
    db = SessionLocal()
    try:
        init_embeddings_table(db)
    except Exception as e:
        print(f"⚠️ DB init warning: {e}")
    finally:
        db.close()

@app.get("/health")
async def health():
    return {"status": "ok", "service": "collabai-ai-service"}