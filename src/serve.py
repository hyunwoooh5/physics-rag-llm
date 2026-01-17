from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from contextlib import asynccontextmanager

from rag import RAGService

from phoenix.otel import register

tracer_provider = register(
    project_name="physics-rag",
    endpoint="http://phoenix:4317",  # Docker Compose
    auto_instrument=True
)


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class APIResponse(BaseModel):
    query: str
    answer: str
    context_used: List[str]


rag_service = RAGService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    '''
    lifespan manager
    '''
    print("Loading RAG resources...")
    try:
        rag_service.load_resources()
        print("RAG resources loaded successfully.")
    except Exception as e:
        print(f"Failed to load RAG resources: {e}")
        raise e

    yield


app = FastAPI(title="Physics RAG API", lifespan=lifespan)


@app.post("/rag")
def query_rag(request: QueryRequest) -> APIResponse:
    result = rag_service.generate_answer(request.query, request.top_k)

    return APIResponse(
        query=request.query,
        answer=result["answer"],
        context_used=result["context_used"]
    )


@app.get("/health")
def health_check():
    return {"status": "running", "rag_ready": rag_service.is_ready}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serve:app", host="0.0.0.0", port=8080, reload=True)
