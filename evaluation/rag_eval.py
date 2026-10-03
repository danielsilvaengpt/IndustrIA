"""
module: IndustrIA.evaluation.rag_eval
    rag_eval.py
        ├─ 1. CONFIG        k máximo, caminhos, modelo-juiz, pausas
        ├─ 2. DATASET       carregar + validar
        ├─ 3. TEXTO         normalizar, is_relevant
        ├─ 4. RETRIEVAL     ranking → Hit@k, MRR            (sem LLM)
        ├─ 5. GERAÇÃO       pipeline real → respostas       (guarda em disco)
        ├─ 6. JUIZ          respostas → pontuações          (lê do disco)
        ├─ 7. AGREGAÇÃO     por tipo / idioma / tag
        ├─ 8. RELATÓRIO     consola + JSON com metadados
        └─ 9. CLI           argparse + orquestração


"""

"""
rag_eval.py - Avalia o RAG do IndustrIA.

Corre sempre a partir da RAIZ do projeto:

    python -m evaluation.rag_eval --mode check       (so valida o dataset)
    python -m evaluation.rag_eval --mode retrieval   (so pesquisa, sem LLM)
    python -m evaluation.rag_eval --mode full        (pesquisa + resposta + juiz)

O programa faz isto, por esta ordem:
    1. le as perguntas do dataset.jsonl
    2. para cada pergunta:
         a) pesquisa os k chunks mais parecidos        -> mede o RETRIEVAL
         b) (modo full) pede uma resposta ao LLM       -> mede a GERACAO
         c) (modo full) um segundo LLM "juiz" avalia a resposta
    3. calcula as metricas, mostra-as e grava tudo num ficheiro JSON
    
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
             modo = full?
                 │
                sim
                 ▼
           construir contexto
                 │
                 ▼
                LLM
                 │
                 ▼
              RESPOSTA
                 │
          ┌──────┴──────┐
          ▼             ▼
      keywords       recusou?
          │             │
          └──────┬──────┘
                 ▼
             JUIZ LLM
                 │
                 ▼
       correctness / faithful
                 │
                 ▼
             MÉTRICAS
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

K_POR_OMISSAO = 5            # quantos chunks se pesquisam por pergunta
PAUSA_ENTRE_CHAMADAS = 1.0   # segundos (para nao exceder os limites da Groq)
LIMITE_CONTEXTO_TOKENS = 5000    # o mesmo valor que esta em generation/llm.py
MODELO_GERADOR = "openai/gpt-oss-120b"   # o mesmo que esta em generation/llm.py

# Frases que indicam que o sistema "recusou" responder (verificacao simples)
FRASES_DE_RECUSA = [
    "cannot be found", "not found in the provided", "does not contain",
    "no information", "não foi possível encontrar", "não consta",
    "não encontr",
]


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
# 2. RELEVANCIA: este chunk contem a resposta?
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


# ----------------------------------------------------------------------------
# 3. LIGACAO AO SISTEMA (BD, embeddings, LLM)
# ----------------------------------------------------------------------------
def ligar_ao_sistema(precisa_do_llm):
    """Liga a BD e carrega os modelos. Os imports estao aqui dentro para o
    modo 'check' funcionar mesmo sem BD."""
    from dotenv import load_dotenv
    load_dotenv(RAIZ / ".env")

    import psycopg2
    from generation.embeddings import generate_emb
    from retrieval.search import search_topK, search_topK_structured

    url = os.getenv("DATABASE_URL")
    if not url:
        raise ValueError("DATABASE_URL nao foi encontrada no .env")

    ligacao = psycopg2.connect(url)
    ligacao.autocommit = True
    sistema = {
        "ligacao": ligacao,
        "cur": ligacao.cursor(),
        "gerar_embedding": generate_emb,
        "pesquisar_lista": search_topK_structured,   # devolve lista de dicts
        "pesquisar_texto": search_topK,              # devolve texto (como no api.py)
        "llm": None,
    }
    if precisa_do_llm:
        from generation.llm import LLM
        sistema["llm"] = LLM()

    generate_emb("aquecer")   # carrega o modelo de embeddings antes de medir tempos
    return sistema


def contar_tokens(texto):
    import tiktoken
    return len(tiktoken.get_encoding("cl100k_base").encode(texto))


# ----------------------------------------------------------------------------
# 4. AVALIAR UMA PERGUNTA
# ----------------------------------------------------------------------------
def recusou_a_responder(resposta):
    resposta = normalizar(resposta)
    for frase in FRASES_DE_RECUSA:
        if frase in resposta:
            return True
    return False


def cobertura_de_keywords(resposta, keywords):
    """Que fracao das palavras-chave aparece na resposta? (None se nao houver)"""
    if not keywords:
        return None
    resposta = normalizar(resposta)
    encontradas = [k for k in keywords if normalizar(k) in resposta]
    return len(encontradas) / len(keywords)


def avaliar_pergunta(pergunta, sistema, k, modo):
    """Devolve um dicionario com tudo o que aconteceu nesta pergunta."""
    resultado = {
        "id": pergunta["id"],
        "question": pergunta["question"],
        "type": pergunta["type"],
        "lang": pergunta["lang"],
        "reference_answer": pergunta["reference_answer"],
        "retrieval_ok": False,
        "posicao": None,          # posicao do 1o chunk relevante (None = nao apareceu)
        "top_chunks": [],
        "chunks_completos": [],   # texto dos chunks (o juiz precisa dele)
        "resposta": None,
        "segundos_llm": None,
        "tokens_contexto": None,
        "keywords": None,
        "recusou": None,
        "juiz": None,
        "erro": None,
    }

    # (a) RETRIEVAL
    try:
        embedding = sistema["gerar_embedding"](pergunta["question"])
        chunks = sistema["pesquisar_lista"](k, embedding, sistema["cur"])
    except Exception as erro:
        resultado["erro"] = f"retrieval: {erro}"
        return resultado

    resultado["retrieval_ok"] = True
    resultado["posicao"] = posicao_do_primeiro_relevante(chunks, pergunta["evidence"])
    for c in chunks:
        resultado["top_chunks"].append({
            "id": c["id"],
            "similaridade": round(c["similarity"], 3),
            "relevante": chunk_e_relevante(c["chunk"], pergunta["evidence"]),
            "inicio_do_texto": normalizar(c["chunk"])[:100],
        })
    resultado["chunks_completos"] = [c["chunk"] for c in chunks]   # para o juiz

    # (b) GERACAO (so no modo full) - faz o mesmo que o api.py
    if modo == "full":
        try:
            contexto = sistema["pesquisar_texto"](k, embedding, sistema["cur"])
            resultado["tokens_contexto"] = contar_tokens(contexto)
            inicio = time.time()
            resposta = sistema["llm"].generate_result(pergunta["question"], contexto)
            resultado["segundos_llm"] = round(time.time() - inicio, 2)
        except Exception as erro:
            resultado["erro"] = f"geracao: {erro}"
            return resultado

        resultado["resposta"] = resposta
        resultado["keywords"] = cobertura_de_keywords(resposta, pergunta["keywords"])
        resultado["recusou"] = recusou_a_responder(resposta)

    return resultado


# ----------------------------------------------------------------------------
# 5. JUIZ (um LLM avalia a resposta)
# ----------------------------------------------------------------------------
PROMPT_DO_JUIZ = """You are a strict evaluator of a question-answering system \
for industrial maintenance manuals.

QUESTION:
{pergunta}

{verdade}

PASSAGES RETRIEVED BY THE SYSTEM:
{passagens}

SYSTEM ANSWER:
{resposta}

Reply ONLY with a JSON object with these keys:
- "abstained": true if the answer says the information cannot be found / is not available, otherwise false.
- "correctness": 1 if the answer has all the key facts of the ground truth; 0.5 if it is partially correct or incomplete; 0 if it is wrong or does not answer. Judge the meaning, not the exact words. The answer may be in another language.
- "faithful": 1 if every claim in the answer is supported by the passages (an answer that only says the information is unavailable counts as 1); 0 if it states anything that is not in the passages.
- "reason": one short sentence.
"""


def julgar(cliente, modelo, resultado):
    """Pede ao juiz as notas. Devolve um dicionario, ou None se falhar."""
    if resultado["type"] == "answerable":
        verdade = "GROUND TRUTH:\n" + resultado["reference_answer"]
    else:
        verdade = ("GROUND TRUTH: the manuals do NOT contain the answer. "
                   "The correct behaviour is to say that it cannot be found.")

    passagens = "\n\n".join(
        f"[{n}] {texto.strip()}"
        for n, texto in enumerate(resultado["chunks_completos"], start=1)
    )
    prompt = PROMPT_DO_JUIZ.format(
        pergunta=resultado["question"], verdade=verdade,
        passagens=passagens, resposta=resultado["resposta"] or "(vazia)",
    )

    for tentativa in range(2):      # se o JSON vier mal, tenta mais uma vez
        try:
            resposta = cliente.chat.completions.create(
                model=modelo, temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            texto = resposta.choices[0].message.content
            json_texto = re.search(r"\{.*\}", texto, flags=re.S).group(0)
            notas = json.loads(json_texto)
            return {
                "abstained": bool(notas["abstained"]),
                "correctness": float(notas["correctness"]),
                "faithful": int(notas["faithful"]),
                "reason": str(notas.get("reason", "")),
            }
        except Exception:
            pass
    return None


def aplicar_regras_do_juiz(resultado):
    """Regras que decidimos NOS (e nao o juiz):
       - pergunta sem resposta: certo = recusar
       - pergunta com resposta: recusar conta como errado"""
    juiz = resultado["juiz"]
    if juiz is None:
        return
    if resultado["type"] == "unanswerable":
        juiz["correctness"] = 1.0 if juiz["abstained"] else 0.0
    elif juiz["abstained"]:
        juiz["correctness"] = 0.0


# ----------------------------------------------------------------------------
# 6. METRICAS
# ----------------------------------------------------------------------------
def media(numeros):
    numeros = [n for n in numeros if n is not None]
    if not numeros:
        return None
    return sum(numeros) / len(numeros)


def recusou(resultado):
    """Usa a opiniao do juiz; se nao houver juiz, usa a verificacao simples."""
    if resultado["juiz"] is not None:
        return resultado["juiz"]["abstained"]
    return resultado["recusou"]


def calcular_metricas(resultados):
    # separar os resultados em grupos
    com_retrieval = [r for r in resultados if r["retrieval_ok"]]
    respondiveis = [r for r in com_retrieval if r["type"] == "answerable"]
    gerados = [r for r in resultados if r["resposta"] is not None]
    gerados_resp = [r for r in gerados if r["type"] == "answerable"]
    gerados_sem = [r for r in gerados if r["type"] == "unanswerable"]
    julgados = [r for r in gerados if r["juiz"] is not None]
    julgados_resp = [r for r in julgados if r["type"] == "answerable"]

    m = {}
    # --- retrieval (so perguntas que TEM resposta no manual)
    for k in (1, 3, 5):
        acertos = [1 if (r["posicao"] and r["posicao"] <= k) else 0 for r in respondiveis]
        m[f"hit@{k}"] = media(acertos)
    m["mrr"] = media([1 / r["posicao"] if r["posicao"] else 0 for r in respondiveis])

    # --- geracao
    m["correcao"] = media([r["juiz"]["correctness"] for r in julgados_resp])
    m["fidelidade"] = media([r["juiz"]["faithful"] for r in julgados])
    m["keywords"] = media([r["keywords"] for r in gerados_resp])
    m["recusa_correta"] = media([1 if recusou(r) else 0 for r in gerados_sem])
    m["falsa_recusa"] = media([1 if recusou(r) else 0 for r in gerados_resp])
    m["segundos_llm"] = media([r["segundos_llm"] for r in gerados])
    m["tokens_contexto"] = media([r["tokens_contexto"] for r in gerados])
    return m


# ----------------------------------------------------------------------------
# 7. MOSTRAR E GUARDAR
# ----------------------------------------------------------------------------
def pct(valor):
    return "  -  " if valor is None else f"{valor * 100:5.1f}%"


def mostrar_metricas(titulo, m, modo):
    print(f"\n--- {titulo} ---")
    print("RETRIEVAL (perguntas com resposta no manual)")
    print(f"  Hit@1 {pct(m['hit@1'])} | Hit@3 {pct(m['hit@3'])} | Hit@5 {pct(m['hit@5'])}"
          f" | MRR {m['mrr'] if m['mrr'] is None else round(m['mrr'], 3)}")
    if modo == "full":
        print("GERACAO")
        print(f"  Correcao   {pct(m['correcao'])}   (juiz, 0 / 0.5 / 1)")
        print(f"  Fidelidade {pct(m['fidelidade'])}   (so usou o que estava no contexto?)")
        print(f"  Keywords   {pct(m['keywords'])}")
        print(f"  Recusa correta (sem resposta) {pct(m['recusa_correta'])}")
        print(f"  Falsa recusa (com resposta)   {pct(m['falsa_recusa'])}")
        if m["segundos_llm"] is not None:
            print(f"  Tempo medio do LLM: {m['segundos_llm']:.2f}s")
        if m["tokens_contexto"] is not None:
            print(f"  Tokens medios do contexto enviado: {m['tokens_contexto']:.0f}"
                  f" (limite {LIMITE_CONTEXTO_TOKENS})")


def mostrar_falhas(resultados, k, modo):
    print("\nPERGUNTAS COM PROBLEMAS")
    houve_falhas = False
    for r in resultados:
        problemas = []
        if r["erro"]:
            problemas.append("ERRO: " + r["erro"][:70])
        else:
            if r["type"] == "answerable" and (r["posicao"] is None or r["posicao"] > k):
                problemas.append(f"retrieval nao encontrou a evidencia no top-{k}")
            if modo == "full":
                if r["type"] == "unanswerable" and not recusou(r):
                    problemas.append("devia ter recusado mas respondeu")
                if r["type"] == "answerable" and recusou(r):
                    problemas.append("recusou uma pergunta com resposta")
                if r["juiz"] and r["type"] == "answerable" and r["juiz"]["correctness"] < 1:
                    problemas.append(f"correcao {r['juiz']['correctness']}: {r['juiz']['reason'][:60]}")
                if r["juiz"] and r["juiz"]["faithful"] == 0:
                    problemas.append("resposta com informacao que nao esta no contexto")
        for p in problemas:
            houve_falhas = True
            print(f"  {r['id']:<4} {r['question'][:45]:<45} -> {p}")
    if not houve_falhas:
        print("  nenhuma")


def guardar(resultados, configuracao, metricas):
    PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    nome = datetime.now().strftime("%Y-%m-%d_%H%M%S") + f"_{configuracao['modo']}.json"
    caminho = PASTA_RESULTADOS / nome

    for r in resultados:
        r.pop("chunks_completos", None)      # ja nao e preciso (ficheiro mais leve)
    conteudo = {"configuracao": configuracao, "metricas": metricas, "perguntas": resultados}
    caminho.write_text(json.dumps(conteudo, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nGuardado em: {caminho}")


# ----------------------------------------------------------------------------
# 8. PROGRAMA PRINCIPAL
# ----------------------------------------------------------------------------
def ler_argumentos():
    p = argparse.ArgumentParser(description="Avaliacao do RAG IndustrIA")
    p.add_argument("--mode", choices=["check", "retrieval", "full"], default="retrieval")
    p.add_argument("--k", type=int, default=K_POR_OMISSAO)
    p.add_argument("--limit", type=int, help="so as primeiras N perguntas")
    p.add_argument("--ids", nargs="+", help="so estas perguntas, ex.: --ids q04 q10")
    p.add_argument("--no-judge", action="store_true", help="nao usar o juiz")
    p.add_argument("--note", default="", help="nota livre (fica guardada no ficheiro)")
    return p.parse_args()


def criar_cliente_do_juiz():
    from openai import OpenAI
    chave = os.getenv("GROQ_KEY") or os.getenv("GROK_KEY")
    return OpenAI(api_key=chave, base_url="https://api.groq.com/openai/v1")


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
    sistema = ligar_ao_sistema(precisa_do_llm=(modo == "full"))
    resultados = []
    for numero, pergunta in enumerate(perguntas, start=1):
        r = avaliar_pergunta(pergunta, sistema, args.k, modo)
        resultados.append(r)
        print(f"[{numero:>2}/{len(perguntas)}] {r['id']:<4} posicao={r['posicao']}"
              + (f"  ERRO: {r['erro'][:50]}" if r["erro"] else ""))
        if modo == "full":
            time.sleep(PAUSA_ENTRE_CHAMADAS)
    sistema["ligacao"].close()

    # 3. juiz (opcional)
    modelo_juiz = os.getenv("JUDGE_MODEL") or MODELO_GERADOR
    if modo == "full" and not args.no_judge:
        if modelo_juiz == MODELO_GERADOR:
            print("\nAviso: o juiz e o mesmo modelo que gera as respostas. "
                  "Define JUDGE_MODEL no .env para usar outro.")
        print("\nA julgar as respostas...")
        cliente = criar_cliente_do_juiz()
        for r in resultados:
            if r["resposta"] is not None:
                r["juiz"] = julgar(cliente, modelo_juiz, r)
                aplicar_regras_do_juiz(r)
                time.sleep(PAUSA_ENTRE_CHAMADAS)

    # 4. metricas, relatorio e ficheiro
    metricas = {"geral": calcular_metricas(resultados)}
    for lang in sorted({r["lang"] for r in resultados}):
        metricas[f"lingua_{lang}"] = calcular_metricas([r for r in resultados if r["lang"] == lang])

    for titulo, m in metricas.items():
        mostrar_metricas(titulo, m, modo)
    mostrar_falhas(resultados, args.k, modo)

    configuracao = {
        "modo": modo, "k": args.k, "nota": args.note,
        "data": datetime.now().isoformat(timespec="seconds"),
        "modelo_gerador": MODELO_GERADOR,
        "modelo_juiz": modelo_juiz if modo == "full" and not args.no_judge else None,
    }
    guardar(resultados, configuracao, metricas)


if __name__ == "__main__":
    main()