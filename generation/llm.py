import os

from dotenv import load_dotenv
from generation.prompt import promptTemplate
from openai import OpenAI
from generation.prompt import truncate_context

load_dotenv()

groq_key = os.getenv("GROQ_KEY") or os.getenv("GROK_KEY")


class LLM:
    def __init__(self):
        if not groq_key:
            raise ValueError("GROQ_KEY não foi encontrada no arquivo .env")

        self.client = OpenAI(
            api_key=groq_key,
            base_url="https://api.groq.com/openai/v1",
        )

    def generate_result(self, query_search: str, topK_context: str) -> str:
        topK_context = truncate_context(topK_context, max_tokens=5000)
        prompt = promptTemplate(query_search, topK_context)
      
        response = self.client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "system",
                    "content": "You answer questions about industrial maintenance manuals.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )

        return response.choices[0].message.content or ""
