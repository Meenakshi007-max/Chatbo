import os
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from litellm import completion
from dotenv import load_dotenv

# =========================
# 🔐 Load ENV
# =========================
load_dotenv()
BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("OPENAI_API_KEY")

# ==========================
# 📦 Load embedding model
# ==========================
embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# =========================
# 📚 Load knowledge chunks
# =========================
INPUT_FOLDER = "outputs"
all_chunks = []

for file in os.listdir(INPUT_FOLDER):
    if file.endswith("_chunks.json"):
        with open(os.path.join(INPUT_FOLDER, file), "r", encoding="utf-8") as f:
            chunks = json.load(f)
            all_chunks.extend(chunks)

print(f"✅ Loaded {len(all_chunks)} knowledge chunks")

# =========================
# 🔢 Create embeddings once
# =========================
texts = [c["text"] for c in all_chunks]
embeddings = embed_model.encode(texts, show_progress_bar=True)


# =========================
# 😊 Greeting detection
# =========================
def is_greeting(text):
    greetings = ["hi", "hello", "hey", "hai", "namaste"]
    return text.lower().strip() in greetings


# =========================
# 🔎 Retrieval
# =========================
def retrieve(query, top_k=3):
    query_vec = embed_model.encode(query).reshape(1, -1)
    sims = cosine_similarity(query_vec, embeddings)[0]

    top_idx = sims.argsort()[-top_k:][::-1]
    return [all_chunks[i]["text"] for i in top_idx]


# =========================
# 🤖 LLM Answer Generator
# =========================
def ask_llm(query, chunks):
    context = "\n\n".join(chunks)

    response = completion(
        model="gemini-pro",
        api_base=BASE_URL,
        api_key=API_KEY,
        custom_llm_provider="openai",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. "
                    "If greeting, respond normally. "
                    "For knowledge questions, use ONLY provided context. "
                    "Do NOT mention context or chunks. "
                    "If answer not present, say 'Answer not found in knowledge base'. "
                    "Answer in same language as question."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}"
            }
        ]
    )

    return response["choices"][0]["message"]["content"]


# =========================
# ▶️ MAIN (Continuous Chat)
# =========================
if __name__ == "__main__":
    print("\n🤖 Chatbot ready! Type 'exit' to stop.\n")

    while True:
        query = input("You: ")

        # Exit condition
        if query.lower() in ["exit", "quit", "bye"]:
            print("🤖 Goodbye!")
            break

        # Greeting shortcut
        if is_greeting(query):
            print("🤖 Hello! How can I help you?\n")
            continue

        # Retrieve KB chunks
        top_chunks = retrieve(query)

        # Generate answer
        answer = ask_llm(query, top_chunks)

        print("\n🤖 Answer:\n")
        print(answer)
        print("\n" + "-"*50 + "\n")