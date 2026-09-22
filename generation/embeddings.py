from data.emb_schema import EmbeddingSchema
from generation.prompt import generate_flatText
from sentence_transformers import SentenceTransformer

model = None


def _get_model():
    global model

    if model is None:
        try:
            model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as error:
            raise RuntimeError(
                "Não foi possível carregar o modelo 'all-MiniLM-L6-v2'. "
                "Verifique a conexão com o Hugging Face ou informe um caminho "
                "local para um modelo já baixado."
            ) from error

    return model


def generate_emb_chunks(chunksList: list[EmbeddingSchema]):
    embedding_model = _get_model()

    for chunk in chunksList:
        chunk.embedding = embedding_model.encode(
            generate_flatText(chunk),
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2 normalization
        ).tolist()
    
    return chunksList


def generate_emb(text: str):
    embedding_model = _get_model()

    emb = embedding_model.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2 normalization              
    ).tolist()
    
    return emb