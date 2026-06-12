from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
import google.generativeai as genai
from PIL import Image

# Gemini API Key
genai.configure(api_key="your_api_key")

model = genai.GenerativeModel("gemini-2.5-flash")

# Load Manual
with open("manuals.txt", "r", encoding="utf-8") as f:
    manual_text = f.read()

chunks = [chunk.strip() for chunk in manual_text.split("\n\n") if chunk.strip()]

# Embeddings
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = embed_model.encode(chunks)

index = faiss.IndexFlatL2(embeddings.shape[1])

index.add(np.array(embeddings, dtype=np.float32))

# Sensor Data
sensor_data = pd.read_csv("sensor_data.csv")

# Machine Image
image = Image.open("machine.jpg")

print("=" * 60)
print(" AI Maintenance Copilot ")
print(" Type 'exit' to quit")
print("=" * 60)

while True:

    query = input("\nTechnician: ")

    if query.lower() == "exit":
        break

    query_embedding = embed_model.encode([query])

    D, I = index.search(
        np.array(query_embedding, dtype=np.float32),
        k=min(2, len(chunks))
    )

    retrieved_manual = "\n".join(
        [chunks[i] for i in I[0]]
    )

    prompt = f"""
You are an AI Maintenance Copilot.

Manual Information:
{retrieved_manual}

Sensor Data:
{sensor_data.to_string(index=False)}

Technician Question:
{query}

Analyze:
1. Manual Information
2. Sensor Data
3. Machine Image

Provide:
- Possible Cause
- Risk Level (Low/Medium/High)
- Recommended Action
- Machine Health Score (0-100)
"""

    response = model.generate_content(
        [prompt, image]
    )

    print("\nMaintenance Report:")
    print(response.text)