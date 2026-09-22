from data.emb_schema import EmbeddingSchema

import tiktoken  # or count words as an approximation

def truncate_context(context: str, max_tokens: int = 5000) -> str:
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(context)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        return enc.decode(tokens)
    return context

def generate_flatText (chunk: EmbeddingSchema) -> str: 
    return f"""
        Manual: {chunk.manual}
        Chunk: {chunk.chunk}
        Page: {chunk.page}
    """.strip()


def promptTemplate(query: str, context: str) -> str:
    return f"""You are an industrial maintenance assistant.

        Answer the question using only the information in the provided context.
        If the context does not contain enough information, say that the answer
        cannot be found in the provided manuals. Do not invent procedures,
        specifications, warnings, or diagnoses.

        Question:
        {query}

        Context from the manuals:
        {context}

        Answer in a clear and concise way. Include the manual name and page when
        that information is available in the context.
    """.strip()