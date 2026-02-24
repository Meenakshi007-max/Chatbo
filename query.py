from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

# Load vector index
with open("vector_index_local.pkl", "rb") as f:
    vector_index = pickle.load(f)

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Query
query = "Explain AI in simple terms"
query_vec = model.encode(query).reshape(1, -1)

# Compare with all stored embeddings
vectors = np.array([v["embedding"] for v in vector_index])
similarities = cosine_similarity(query_vec, vectors)[0]

# Get top 3
top_idx = similarities.argsort()[-3:][::-1]

for idx in top_idx:
    print(f"Score: {similarities[idx]:.4f}")
    print("Text:", vector_index[idx]["text"])
    print("---")