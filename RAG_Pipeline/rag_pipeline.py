import faiss
import numpy as np

documents = ["What is AI?", "How does machine learning work?", "What is a neural network?"]
vector_store = faiss.IndexFlatL2(768)

for doc in documents:
    embedding = np.random.rand(768).astype('float32')
    vector_store.add(np.array([embedding]))

def search(query):
    query_embedding = np.random.rand(768).astype('float32')
    D, I = vector_store.search(np.array([query_embedding]), k=1)
    return documents[I[0][0]]

print(search("Tell me about AI"))

