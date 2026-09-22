# IndustriIA

Sistema de perguntas e respostas sobre manuais técnicos de manutenção industrial, baseado numa arquitectura **RAG** (*Retrieval-Augmented Generation*).

O projecto transforma manuais em PDF numa base de conhecimento pesquisável. Para cada pergunta, o sistema cria um embedding, recupera os excertos mais semelhantes e utiliza um modelo de linguagem para formular uma resposta fundamentada no conteúdo recuperado.

> **Estado:** protótipo em desenvolvimento. O projecto está preparado para evoluir para uma API e para suportar mais manuais e fontes de informação.

## Funcionalidades

- extracção de texto a partir de ficheiros PDF;
- divisão do conteúdo em blocos com sobreposição;
- criação de embeddings com `all-MiniLM-L6-v2`;
- armazenamento dos embeddings em PostgreSQL com `pgvector`;
- pesquisa por similaridade de cosseno;
- construção de prompts com contexto recuperado;
- geração de respostas através da API da Groq;
- limite de contexto para controlar o tamanho do pedido ao modelo.

## Arquitectura

```text
PDF
  |
  v
Extracção de texto -> Chunking -> Embeddings -> PostgreSQL + pgvector
                                                    |
Pergunta -> Embedding da pergunta -> Retrieval ----+
                                                    |
                                                    v
                                       Prompt com contexto -> LLM -> Resposta
```

## Estrutura do projecto

```text
IndustrIA/
├── API/
│   └── api.py                 # Ponto de entrada FastAPI (em evolução)
├── data/
│   └── emb_schema.py          # Schema e operações da tabela de embeddings
├── generation/
│   ├── embeddings.py          # Criação dos embeddings
│   ├── llm.py                 # Cliente da API da Groq
│   └── prompt.py              # Prompt e preparação do contexto
├── ingestion/
│   ├── chunking.py            # Divisão do texto em chunks
│   └── pdf_extraction.py      # Leitura de PDFs
├── retrieval/
│   └── search.py              # Pesquisa dos chunks mais semelhantes
├── data/manuals/              # Manuais PDF locais
├── main.py                    # Execução do pipeline
└── README.md
```

## Requisitos

- Python 3.11 ou superior;
- PostgreSQL 17;
- extensão [`pgvector`](https://github.com/pgvector/pgvector) instalada no servidor PostgreSQL;
- uma chave de API da Groq;
- acesso à Internet na primeira utilização, para descarregar o modelo de embeddings.

## Instalação

Na raiz do projecto, crie e active um ambiente virtual:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Instale as dependências do projecto:

```powershell
pip install -r requirements.txt
```

## Configuração

Crie um ficheiro `.env` na raiz do projecto:

```env
DATABASE_URL=postgresql://utilizador:palavra-passe@localhost:5432/nome_da_base_de_dados
GROQ_KEY=chave_da_api
```

O ficheiro `.env` contém informação sensível e não deve ser enviado para o GitHub.

Depois de instalar o `pgvector`, active a extensão na base de dados usada pelo projecto:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

A tabela de embeddings utiliza vectores com 384 dimensões, compatíveis com `all-MiniLM-L6-v2`.

## Execução

Coloque os manuais PDF em `data/manuals/` e execute:

```powershell
python .\main.py
```

O pipeline irá:

1. ler os PDFs existentes;
2. dividir o texto em chunks;
3. gerar embeddings;
4. guardar os chunks na base de dados;
5. gerar o embedding da pergunta;
6. recuperar os resultados mais semelhantes;
7. enviar a pergunta e o contexto para o modelo de linguagem.

## API

Existe um ponto de entrada inicial em `API/api.py`. Para o executar com Uvicorn:

```powershell
uvicorn API.api:app --reload
```

Endpoints disponíveis:

- `GET /` — verifica se a API está disponível;
- `GET /search_query?search_query=...&k=5` — pesquisa os manuais e devolve uma resposta gerada com base no contexto recuperado.

Exemplo:

```text
http://127.0.0.1:8000/search_query?search_query=What%20safety%20procedures%20are%20recommended%20for%20heavy%20fuel%20oil%3F&k=5
```

## Notas importantes

- A instalação da extensão `pgvector` é feita no servidor PostgreSQL, não através do `pip`.
- A tabela deve utilizar `vector(384)` quando os embeddings forem gerados com `all-MiniLM-L6-v2`.
- O conteúdo enviado ao modelo deve ser tratado como contexto de referência; o sistema deve declarar quando os manuais não contêm informação suficiente.
- Os PDFs, credenciais, ambientes virtuais e caches locais não devem ser versionados.

## Próximos passos

- separar ingestão e consulta em comandos distintos;
- criar endpoints de perguntas na API;
- acrescentar citações estruturadas com manual e página;
- adicionar testes automatizados para ingestão, retrieval e prompts;
- incluir histórico de manutenção como fonte adicional;
- preparar execução com Docker para simplificar o PostgreSQL e o `pgvector`.

## Licença

Ainda não definida.