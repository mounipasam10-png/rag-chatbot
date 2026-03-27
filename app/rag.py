import faiss
import numpy as np
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

documents = []
vectors = []

def load_data():
    global documents, vectors
    with open("data/sample.txt", "r") as f:
        docs = f.read().split("\n")

    documents = [d for d in docs if d.strip()]

    embeddings = []
    for doc in documents:
        emb = client.embeddings.create(
            model="text-embedding-3-small",
            input=doc
        )
        embeddings.append(emb.data[0].embedding)

    vectors = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(len(vectors[0]))
    index.add(vectors)

    return index

index = load_data()

def retrieve(query):
    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    query_vector = np.array([emb.data[0].embedding]).astype("float32")

    D, I = index.search(query_vector, k=2)

    return [documents[i] for i in I[0]]
