import os

import psycopg2
from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/search_query")
async def search(search_query: str, k: int = 5):
    from generation.embeddings import generate_emb
    from generation.llm import LLM
    from retrieval.search import search_topK

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL não foi encontrada no ficheiro .env")

    if k < 1 or k > 20:
        raise ValueError("k deve estar entre 1 e 20")

    connection = psycopg2.connect(database_url)
    try:
        with connection.cursor() as cursor:
            context = search_topK(k, generate_emb(search_query), cursor)

        answer = LLM().generate_result(search_query, context)
        return {
            "query": search_query,
            "answer": answer,
            "results": k,
        }
    finally:
        connection.close()
