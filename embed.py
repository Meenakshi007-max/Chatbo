import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer

INPUT_FOLDER = "outputs"  # where your chunk JSONs are
VECTOR_FILE = "vector_index_local.pkl"  # final output

# Load local embedding model
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")  # offline model, ~384 dims

vector_index = []

# Loop over all chunk files
json_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith("_chunks.json")]

for json_file in json_files:
    with open(os.path.join(INPUT_FOLDER, json_file), "r", encoding="utf-8") as f:
        chunks = json.load(f)

    for chunk in chunks:
        text = chunk["text"]
        vector = model.encode(text)  # vector is a numpy array
        chunk["embedding"] = vector.tolist()  # convert to list for saving

        vector_index.append(chunk)

        # Optional: print vector info
        print(f"Chunk {chunk['chunk_id']} vector length:", len(vector))
        print("First 5 values:", vector[:5])

# Save all vectors locally
import pickle
with open(VECTOR_FILE, "wb") as f:
    pickle.dump(vector_index, f)

print(f"✅ Local vector index saved to {VECTOR_FILE}")