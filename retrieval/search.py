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

