# CollabAI AI Service

A Python FastAPI microservice providing AI capabilities for the CollabAI platform.

## Tech Stack

- **Python 3.11** + **FastAPI**
- **OpenAI GPT-4o-mini** (LLM)
- **OpenAI text-embedding-3-small** (embeddings)
- **PostgreSQL** (embedding storage)
- **RAG** (Retrieval-Augmented Generation)
- **Uvicorn** (ASGI server)

## Features

- Document summarization
- Text rewriting in 5 modes (improve, formal, casual, shorter, longer)
- RAG-powered document Q&A
- Semantic search using cosine similarity
- Auto-chunking with overlap for better context

## RAG Pipeline
Document saved
→ chunk into ~500 char segments with overlap
→ embed each chunk via OpenAI
→ store in PostgreSQL as JSON vectors
User asks question
→ embed question
→ cosine similarity search against all chunks
→ retrieve top-4 most relevant chunks
→ send to GPT-4o-mini with context
→ return grounded answer

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /api/ai/summarize | Summarize document |
| POST | /api/ai/rewrite | Rewrite text |
| POST | /api/ai/chat | Chat with document (RAG) |
| POST | /api/ai/embed | Embed document chunks |
| GET | /health | Health check |

## Running Locally

### Prerequisites
- Python 3.11+
- PostgreSQL 17
- OpenAI API key

### Setup

1. Create virtual environment:
```bash
python -m venv venv
venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create `.env` file (see `.env.example`):
OPENAI_API_KEY=sk-...
DATABASE_URL=postgresql://postgres:password@localhost:5432/collabai_db

4. Run:
```bash
uvicorn app.main:app --reload --port 8000
```

API docs available at `http://localhost:8000/docs`

## Running with Docker

```bash
docker-compose up --build
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| OPENAI_API_KEY | OpenAI API key |
| DATABASE_URL | PostgreSQL connection URL |

## Related Repositories

- [collabai-backend](https://github.com/raventext/collabai-backend) — Spring Boot backend
- [collabai-frontend](https://github.com/raventext/collabai-frontend) — React frontend
