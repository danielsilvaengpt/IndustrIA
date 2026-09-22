class EmbeddingSchema:

    def __init__(self):
        self.manual = ""
        self.chunk = ""
        self.embedding = []
        self.page = 0

    def set_values(
        self,
        manual: str,
        chunk: str,
        page: int = 0,
        embedding: list[float] | None = None
    ):
        self.manual = manual
        self.chunk = chunk
        self.embedding = embedding if embedding is not None else []
        self.page = page

    def to_dict(self):
        return {
            "manual": self.manual,
            "chunk": self.chunk,
            "embedding": self.embedding,
            "page": self.page
        }

    def create_table(self):
        return """
        CREATE EXTENSION IF NOT EXISTS vector;
        CREATE SCHEMA IF NOT EXISTS embeddings_schema;

        CREATE TABLE IF NOT EXISTS embeddings_schema.embeddings (
            id SERIAL PRIMARY KEY,
            manual TEXT NOT NULL,
            chunk TEXT NOT NULL,
            embedding vector(384),
            page INT NOT NULL
        );
        """

    def insert_embedding(self):
        return """
            INSERT INTO embeddings_schema.embeddings
            (manual, chunk, embedding, page)
            VALUES (%s, %s, %s, %s);
        """, (
            self.manual,
            self.chunk,
            self.embedding if self.embedding else None,
            self.page
        )
        
    def select_all(self):
        return """
                    SELECT * FROM embeddings_schema.embeddings;
                """