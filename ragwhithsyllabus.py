from mistralai import Mistral
import requests
import numpy as np
import faiss
import os
import time
from pathlib import Path

# Initialize API client
api_key = "WKiwdDUBsOdhrspodlAOKRJJMLSMw3sy"
client = Mistral(api_key=api_key)

# Fetch text data

from PyPDF2 import PdfReader

file_path = "Syllabus_Ing_2024_2025.pdf"
reader = PdfReader(file_path)
text = ""
for page in reader.pages:
    text += page.extract_text()
    
# Split text into chunks
chunk_size = 1000
chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

### 🔹 FIXED get_text_embedding FUNCTION
def get_text_embedding(inputs):
    time.sleep(2)  # Prevent hitting rate limits
    response = client.embeddings.create(model="mistral-embed", inputs=inputs)
    return [embedding.embedding for embedding in response.data]
       
### 🔹 BATCHED REQUESTS
batch_size = 100  # Adjust based on API limits
text_embeddings = []

for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i + batch_size]
    print(f"Processing batch {i//batch_size + 1}/{len(chunks)//batch_size + 1}...")
    text_embeddings.extend(get_text_embedding(batch))

text_embeddings = np.array(text_embeddings)

# Build FAISS index
d = text_embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(text_embeddings)

time.sleep(2)
# Run Mistral
def run_mistral(question, model="mistral-large-latest"):
    # Query processing
    time.sleep(2)
    question_embeddings = np.array(get_text_embedding([question]))  # Batch single query

    D, I = index.search(question_embeddings, k=4)  # Retrieve top 2 relevant chunks
    retrieved_chunk = [chunks[i] for i in I[0]]

    messages = [{"role": "user", "content":
    f"""
Context information is below.
---------------------
{retrieved_chunk}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {question}
Answer:
""" }]
    time.sleep(2)
    chat_response = client.chat.complete(model=model, messages=messages)
    return (chat_response.choices[0].message.content)

print(run_mistral("Est ce que Alain KILIDJIAN est professeur d'un élèctif?"))
