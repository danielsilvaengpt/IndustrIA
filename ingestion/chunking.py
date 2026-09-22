from data.emb_schema import EmbeddingSchema


def chunk_text(pdf_info: dict, chunk_size: int = 1000, overlap: int = 50):
    """
    Splits the input text into chunks of specified size.

    Args:
        pdf_info (dict): A dictionary containing the PDF name and text.
        chunk_size (int): The maximum size of each chunk.
        overlap (int): The number of characters to overlap between chunks.

    Returns:
        list: A list containing the chunk objects.
    """

    text = pdf_info.get("text", "")
    if not text:
        return []

    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 2)

    chunks = []
    start = 0

    while start < len(text):
        print(f"Processing chunk at position {start}")

        end = min(start + chunk_size, len(text))
        end_point = text.rfind("\n", start, end)

        if end_point != -1 and end_point > start:
            chunk = text[start:end_point + 1]
        else:
            chunk = text[start:end]

        emb = EmbeddingSchema()
        emb.set_values(
            manual=pdf_info["name"],
            chunk=chunk,
            page=len(chunks) + 1
        )
        chunks.append(emb)

        next_start = start + max(1, len(chunk) - overlap)
        if next_start <= start:
            break

        start = min(next_start, len(text))

    return chunks