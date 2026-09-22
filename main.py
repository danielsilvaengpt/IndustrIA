from data.emb_schema import EmbeddingSchema
import os
import psycopg2
from dotenv import load_dotenv
from pathlib import Path
from ingestion.pdf_extraction import extract_text_from_pdf
from ingestion.chunking import chunk_text
from generation.embeddings import generate_emb_chunks, generate_emb
from retrieval.search import search_topK
from generation.llm import LLM
# Load environment variables from .env file
load_dotenv()

# Connect to the database
db = os.getenv("DATABASE_URL")
if not db:
    raise ValueError("DATABASE_URL não foi encontrada. Verifique o arquivo .env na raiz do projeto.")

db_con = psycopg2.connect(db)

if db_con:
    print("Database connection successful.")
    
cur = db_con.cursor()

# Create an instance of the embedding_schema class
emb = EmbeddingSchema()

# Create the embeddings table if it doesn't exist
cur.execute(emb.create_table())
cur.execute("COMMIT;")


# extract text from pdf
pdf_folder = Path(r"data\manuals")

"""for pdf_path in pdf_folder.glob("*.pdf"):
    result = extract_text_from_pdf(pdf_path)
    print("Texto extraido com sucesso")
    chunks = chunk_text(result)
    print("Chunks extraidas com sucesso")
    chunks = generate_emb_chunks(chunks)
    for c in chunks:
        print(c.to_dict())
        query, values = c.insert_embedding()
        cur.execute(query, values)
    cur.execute("COMMIT;")"""
    
query = "What safety procedures and maintenance practices are recommended for handling heavy fuel oil?"

topK_results = search_topK(5 , generate_emb(query), cur)

llm = LLM()
resultado = llm.generate_result(query , topK_results)
print(f"\n\nResultado:{resultado}")
print("feito!!")

