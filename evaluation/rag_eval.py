"""
module: IndustrIA.evaluation.rag_eval
    rag_eval.py
        ├─ 1. CONFIG        k máximo e caminhos
        ├─ 2. DATASET       carregar + validar
        ├─ 3. TEXTO         normalizar, is_relevant
        ├─ 4. RETRIEVAL     ranking → Recall@k, MRR, latência
        ├─ 5. RELATÓRIO     consola + JSON com metadados
        └─ 6. CLI           argparse + orquestração


"""

"""
rag_eval.py - Avalia o RAG do IndustrIA.

Corre sempre a partir da RAIZ do projeto:

    python -m evaluation.rag_eval --mode check       (so valida o dataset)
    python -m evaluation.rag_eval --k 5              (Recall@5, MRR e latência)

O programa faz isto, por esta ordem:
    1. le as perguntas do dataset.jsonl
    2. para cada pergunta:
            a) pesquisa os k chunks mais parecidos        -> mede o RETRIEVAL
        3. calcula Recall@k, MRR e latência, e grava JSON
    
Estrutura do .py: 

                 dataset.jsonl
                      │
                      ▼
              carregar_dataset()
                      │
                      ▼
              ┌───────────────┐
              │   PERGUNTA    │
              └───────┬───────┘
                      │
                      ▼
               gerar_embedding()
                      │
                      ▼
                pesquisar BD
                      │
                      ▼
                 TOP K CHUNKS
                      │
                      ▼
             evidencia encontrada?
                  /         \
                sim          não
                 │            │
                 ▼            ▼
              Retrieval     Falha
                 │
                 ▼
                 ▼
          Recall@k / MRR
                 │
        ┌────────┴────────┐
        ▼                 ▼
     terminal          JSON

"""
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

# Para conseguir importar as pastas do projeto (generation, retrieval, ...)
RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))


# ----------------------------------------------------------------------------
# CONFIGURACAO
# ----------------------------------------------------------------------------
CAMINHO_DATASET = RAIZ / "evaluation" / "dataset.jsonl"
PASTA_RESULTADOS = RAIZ / "evaluation" / "results"

K_POR_OMISSAO = 5


# ----------------------------------------------------------------------------
# 1. DATASET
# ----------------------------------------------------------------------------
def carregar_dataset(caminho):
    """Le o ficheiro .jsonl: cada linha e uma pergunta (um dicionario)."""
    perguntas = []
    with open(caminho, encoding="utf-8") as ficheiro:
        for linha in ficheiro:
            if linha.strip():
                perguntas.append(json.loads(linha))

    for p in perguntas:
        if p["type"] == "answerable" and not p["evidence"]:
            raise ValueError(f"A pergunta {p['id']} nao tem evidencia.")
    return perguntas


# ----------------------------------------------------------------------------
# 2. RELEVANCIA: este chunk contem uma evidencia?
# ----------------------------------------------------------------------------
def normalizar(texto):
    """Poe o texto num formato comparavel: minusculas e sem quebras de linha."""
    texto = texto.lower()
    texto = texto.replace("’", "'")                  # aspas curvas -> retas
    texto = re.sub(r"(\w)-\s+(\w)", r"\1-\2", texto)  # "ten-\nstep" -> "ten-step"
    texto = re.sub(r"\s+", " ", texto)                # varios espacos/\n -> 1 espaco
    return texto.strip()


def chunk_e_relevante(texto_do_chunk, evidencias):
    """Um chunk e relevante se contiver QUALQUER uma das evidencias."""
    chunk = normalizar(texto_do_chunk)
    for evidencia in evidencias:
        if normalizar(evidencia) in chunk:
            return True
    return False


def posicao_do_primeiro_relevante(chunks, evidencias):
    """Devolve 1, 2, 3... (posicao no ranking) ou None se nenhum for relevante."""
    for posicao, chunk in enumerate(chunks, start=1):
        if chunk_e_relevante(chunk["chunk"], evidencias):
            return posicao
    return None


def recall_no_top_k(chunks, evidencias):
    """Fracao das evidencias encontradas nos chunks recuperados."""
    if not evidencias:
        return None
    encontradas = sum(
        any(normalizar(evidencia) in normalizar(chunk["chunk"]) for chunk in chunks)
        for evidencia in evidencias
    )
    return encontradas / len(evidencias)


# ----------------------------------------------------------------------------
# 3. LIGACAO AO SISTEMA (BD, embeddings, LLM)
# ----------------------------------------------------------------------------
def ligar_ao_sistema():
    """Liga a BD e carrega os modelos. Os imports estao aqui dentro para o
    modo 'check' funcionar mesmo sem BD."""
    from dotenv import load_dotenv
    load_dotenv(RAIZ / ".env")

    import psycopg2
    from generation.embeddings import generate_emb
    from retrieval.search import search_topK_structured

    url = os.getenv("DATABASE_URL")
    if not url:
        raise ValueError("DATABASE_URL nao foi encontrada no .env")

    ligacao = psycopg2.connect(url)
    ligacao.autocommit = True
    sistema = {
        "ligacao": ligacao,
        "cur": ligacao.cursor(),
        "gerar_embedding": generate_emb,
        "pesquisar": search_topK_structured,
    }
    generate_emb("aquecer")
    return sistema


# ----------------------------------------------------------------------------
# 4. AVALIAR UMA PERGUNTA
# ----------------------------------------------------------------------------
def avaliar_pergunta(pergunta, sistema, k):
    """Executa embedding e busca; mede latencia, recall@k e posicao relevante."""
    resultado = {
        "id": pergunta["id"],
        "question": pergunta["question"],
        "type": pergunta["type"],
        "lang": pergunta["lang"],
        "retrieval_ok": False,
        "recall_at_k": None,
        "posicao_primeiro_relevante": None,
        "latencia_segundos": None,
        "top_chunks": [],
        "erro": None,
    }

    inicio = time.perf_counter()
    try:
        embedding = sistema["gerar_embedding"](pergunta["question"])
        chunks = sistema["pesquisar"](k, embedding, sistema["cur"])
    except Exception as erro:
        resultado["erro"] = f"retrieval: {erro}"
        resultado["latencia_segundos"] = round(time.perf_counter() - inicio, 4)
        return resultado

    resultado["latencia_segundos"] = round(time.perf_counter() - inicio, 4)
    resultado["retrieval_ok"] = True
    resultado["recall_at_k"] = recall_no_top_k(chunks, pergunta["evidence"])
    resultado["posicao_primeiro_relevante"] = posicao_do_primeiro_relevante(
        chunks, pergunta["evidence"]
    )
    for c in chunks:
        resultado["top_chunks"].append({
            "id": c["id"],
            "similaridade": round(c["similarity"], 3),
            "relevante": chunk_e_relevante(c["chunk"], pergunta["evidence"]),
            "inicio_do_texto": normalizar(c["chunk"])[:100],
        })
    return resultado


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# 5. METRICAS
# ----------------------------------------------------------------------------
def media(numeros):
    numeros = [n for n in numeros if n is not None]
    if not numeros:
        return None
    return sum(numeros) / len(numeros)


def calcular_metricas(resultados):
    retrieval_ok = [r for r in resultados if r["retrieval_ok"]]
    respondiveis = [
        r for r in retrieval_ok
        if r["type"] == "answerable" and r["recall_at_k"] is not None
    ]
    return {
        "recall_at_k": media([r["recall_at_k"] for r in respondiveis]),
        "mrr": media([
            1 / r["posicao_primeiro_relevante"]
            if r["posicao_primeiro_relevante"] else 0
            for r in respondiveis
        ]),
        "latencia_media_segundos": media(
            [r["latencia_segundos"] for r in resultados]
        ),
        "similaridade_media": media([
            chunk["similaridade"]
            for resultado in retrieval_ok
            for chunk in resultado["top_chunks"]
        ]),
        "perguntas_respondiveis_avaliadas": len(respondiveis),
        "perguntas_com_retrieval_ok": len(retrieval_ok),
    }


# ----------------------------------------------------------------------------
# 6. MOSTRAR E GUARDAR
# ----------------------------------------------------------------------------
def pct(valor):
    return "  -  " if valor is None else f"{valor * 100:5.1f}%"


def mostrar_metricas(m, k):
    print("\n--- MÉTRICAS GERAIS ---")
    print(f"  Recall@{k}: {pct(m['recall_at_k'])}")
    print(f"  MRR: {m['mrr'] if m['mrr'] is None else round(m['mrr'], 3)}")
    if m["similaridade_media"] is not None:
        print(f"  Similaridade média dos chunks recuperados: {m['similaridade_media']:.3f}")
    if m["latencia_media_segundos"] is not None:
        print(f"  Latência média: {m['latencia_media_segundos']:.4f}s por pergunta")
    print(f"  Perguntas respondíveis avaliadas: {m['perguntas_respondiveis_avaliadas']}")
    print(f"  Perguntas com retrieval concluído: {m['perguntas_com_retrieval_ok']}")


def mostrar_falhas(resultados):
    print("\nPERGUNTAS COM PROBLEMAS")
    houve_falhas = False
    for r in resultados:
        problemas = []
        if r["erro"]:
            problemas.append("ERRO: " + r["erro"][:70])
        elif r["type"] == "answerable" and r["recall_at_k"] < 1:
            problemas.append(f"Recall@k incompleto: {pct(r['recall_at_k'])}")
        for p in problemas:
            houve_falhas = True
            print(f"  {r['id']:<4} {r['question'][:45]:<45} -> {p}")
    if not houve_falhas:
        print("  nenhuma")


def guardar(resultados, configuracao, metricas):
    PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    nome = datetime.now().strftime("%Y-%m-%d_%H%M%S") + f"_{configuracao['modo']}.json"
    caminho = PASTA_RESULTADOS / nome

    conteudo = {"configuracao": configuracao, "metricas": metricas, "perguntas": resultados}
    caminho.write_text(json.dumps(conteudo, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nGuardado em: {caminho}")


# ----------------------------------------------------------------------------
# 7. PROGRAMA PRINCIPAL
# ----------------------------------------------------------------------------
def ler_argumentos():
    p = argparse.ArgumentParser(description="Avalia retrieval: Recall@k, MRR e latencia")
    p.add_argument("--mode", choices=["check", "retrieval"], default="retrieval")
    p.add_argument("--k", type=int, default=K_POR_OMISSAO, help="numero de chunks recuperados")
    p.add_argument("--limit", type=int, help="so as primeiras N perguntas")
    p.add_argument("--ids", nargs="+", help="so estas perguntas, ex.: --ids q04 q10")
    p.add_argument("--note", default="", help="nota livre (fica guardada no ficheiro)")
    return p.parse_args()


def main():
    args = ler_argumentos()
    modo = args.mode

    # 1. dataset
    perguntas = carregar_dataset(CAMINHO_DATASET)
    if args.ids:
        perguntas = [p for p in perguntas if p["id"] in args.ids]
    if args.limit:
        perguntas = perguntas[:args.limit]
    n_resp = sum(1 for p in perguntas if p["type"] == "answerable")
    print(f"Dataset: {len(perguntas)} perguntas "
          f"({n_resp} com resposta, {len(perguntas) - n_resp} sem resposta)")
    if modo == "check":
        return

    # 2. avaliar cada pergunta
    sistema = ligar_ao_sistema()
    resultados = []
    for numero, pergunta in enumerate(perguntas, start=1):
        r = avaliar_pergunta(pergunta, sistema, args.k)
        resultados.append(r)
        similaridades = [chunk["similaridade"] for chunk in r["top_chunks"]]
        print(f"[{numero:>2}/{len(perguntas)}] {r['id']:<4} "
              f"recall@{args.k}={r['recall_at_k']} "
              f"similaridades={similaridades} "
              f"latencia={r['latencia_segundos']}s"
              + (f"  ERRO: {r['erro'][:50]}" if r["erro"] else ""))
    sistema["ligacao"].close()

    # metricas, relatorio e ficheiro
    metricas = calcular_metricas(resultados)
    mostrar_metricas(metricas, args.k)
    mostrar_falhas(resultados)

    configuracao = {
        "modo": modo, "k": args.k, "nota": args.note,
        "data": datetime.now().isoformat(timespec="seconds"),
    }
    guardar(resultados, configuracao, metricas)


if __name__ == "__main__":
    main()
