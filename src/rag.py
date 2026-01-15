import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from fastembed import TextEmbedding
from google import genai
from google.genai.types import GenerateContentConfig
from pydantic import BaseModel, Field

load_dotenv()

# configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "physics_rag_collection_lat_nuc"
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
GENAI_MODEL_NAME = "models/gemini-2.5-flash-lite"
THREADS = 4

# Internal schema for Gemini
class RAGOutputSchema(BaseModel):
    answer: str = Field(..., description="Answer for the question based on context (It can include LaTeX equations)")


class RAGService:
    def __init__(self):
        self.qdrant_client = None
        self.embedding_model = None
        self.genai_client = None
        self.is_ready = False

    def load_resources(self):
        """
        Initialize heavy models and clients.
        """

        self.qdrant_client = QdrantClient(QDRANT_URL)
        self.embedding_model = TextEmbedding(
            model_name=EMBEDDING_MODEL_NAME, threads=THREADS)

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found in environment variables.")
        self.genai_client = genai.Client(api_key=api_key)

        self.is_ready = True

    def _get_query_vector(self, text: str) -> List[float]:
        return next(self.embedding_model.embed([text])).tolist()

    def _retrieve_context(self, query: str, top_k: int = 5) -> List[str]:
        query_vector = self._get_query_vector(query)

        search_result = self.qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            using="default",
            limit=top_k,
            with_payload=True
        )

        contexts = []
        for hit in search_result.points:
            if hit.payload:
                title = hit.payload.get("title", "Unknown title")
                preprint_date = hit.payload.get("preprint_date", "None")
                abstract = hit.payload.get("abstract", "No abstract")

                formatted_text = f"Title: {title} ({preprint_date})\nAbstract: {abstract}"
                contexts.append(formatted_text)

        return contexts

    def generate_answer(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        if not self.is_ready:
            raise RuntimeError(
                "RAG service is not initialized. Call load_resources() first")

        context_list = self._retrieve_context(query, top_k)

        context_text = "\n\n---\n\n".join(context_list)

        prompt_template = """
        You are an expert theoretical physicist assisting a junior researcher. 
        Use the following pieces of retrieved context to answer the question. 
        If the answer is not in the context, just say that you don't know based on the provided documents.

        Question: {query}

        Context (Retrieved Papers): {context_text}

        """.strip()

        prompt = prompt_template.format(query=query, context_text=context_text)

        response = self.genai_client.models.generate_content(
            model=GENAI_MODEL_NAME,
            contents=prompt,
            config=GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=RAGOutputSchema
            )
        )

        if not response.text:
            raise RuntimeError("Empty response from Gemini LLM")

        parsed_output = json.loads(response.text)

        return {
            "answer": parsed_output["answer"],
            "context_used": context_list
        }
