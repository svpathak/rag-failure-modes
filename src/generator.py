import os
import time
from dotenv import load_dotenv
from groq import Groq, RateLimitError
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import GROQ_MODEL, TEMPERATURE, MAX_OUTPUT_TOKENS

load_dotenv()


def get_llm_client():
    return Groq(api_key=os.getenv("GROQ_API_KEY"))


def generate(question, retrieved_chunks, client):
    context = "\n\n".join([c["text"] for c in retrieved_chunks])
    prompt = f"""You are a research assistant. Answer the question based only on the provided context.
If the answer is not in the context, say 'I don't know'.
Keep the answer concise.

Context:
{context}

Question: {question}
Answer:"""

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=MAX_OUTPUT_TOKENS,
                temperature=TEMPERATURE,
            )
            time.sleep(20)  # respect # ~7.5 RPM limit
            return response.choices[0].message.content.strip()
        except RateLimitError as e:
            wait = 60 * (attempt + 1)
            print(f"[RATE LIMIT] Waiting {wait}s before retry...")
            time.sleep(wait)
        except Exception as e:
            raise e

    return ""