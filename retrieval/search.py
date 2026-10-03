def search_topK(k: int, query_emb: list[float], cur):
   query_vector = "[" + ",".join(str(value) for value in query_emb) + "]"

   cur.execute(
      """
      WITH query_embedding AS (
         SELECT %s::vector AS embedding
      )
      SELECT
         e.*,
         1 - (e.embedding <=> q.embedding) AS cosine_similarity
      FROM embeddings_schema.embeddings e
      CROSS JOIN query_embedding q
      ORDER BY e.embedding <=> q.embedding
      LIMIT %s;
      """,
      (query_vector, k),
   )

   columns = [description[0] for description in cur.description]
   rows = cur.fetchall()
   return "\n".join(str(dict(zip(columns, row))) for row in rows)



def search_topK_structured(k: int, query_emb: list[float], cur) -> list[dict]:
    """Igual ao search_topK, mas devolve uma LISTA de dicts (ordenada por
    similaridade) e SEM a coluna embedding. Usada pela avaliacao."""
    query_vector = "[" + ",".join(str(value) for value in query_emb) + "]"

    cur.execute(
        """
        SELECT id, manual, chunk, page,
               1 - (embedding <=> %s::vector) AS similarity
        FROM embeddings_schema.embeddings
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
        """,
        (query_vector, query_vector, k),
    )

    return [
        {"id": row[0], "manual": row[1], "chunk": row[2],
         "page": row[3], "similarity": float(row[4])}
        for row in cur.fetchall()
    ]
