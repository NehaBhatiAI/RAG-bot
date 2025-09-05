import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import json
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm_model = "llama-3.3-70b-versatile"
API_URL = "https://api.groq.com/openai/v1/chat/completions"

""" Step 3: Load the Embedding Model """
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embedding_dim = embedding_model.get_sentence_embedding_dimension()  # Typically 384

""" Step 4: Prepare Your Documents """
documents = [
    "The capital of France is Paris.",
    "Python is a popular programming language.",
    "Machine learning is a subset of artificial intelligence.",
    "The Eiffel Tower is in Paris.",
    "Groq provides fast AI inference."
]

""" Step 5: Generate Embeddings for Documents """
doc_embeddings = embedding_model.encode(documents)
doc_embeddings = np.array(doc_embeddings).astype('float32')

""" Step 6: Set Up FAISS Vector Database """
index = faiss.IndexFlatL2(embedding_dim)
index.add(doc_embeddings)

""" Step 7: Define Retrieval and Response Functions """

def retrieve_documents(query, top_k=3):
    query_embedding = embedding_model.encode([query])
    query_embedding = np.array(query_embedding).astype('float32')
    distances, indices = index.search(query_embedding, top_k)
    retrieved_docs = [documents[i] for i in indices[0]]
    return retrieved_docs

def generate_response(query, retrieved_docs, temperature=0.4):
    context = "\n".join([f"- {doc}" for doc in retrieved_docs])

    prompt = f"""
    You are a concise and accurate assistant. Use the provided context to answer the query directly and clearly. 
    If the context doesn't contain relevant information, then simply say: **Can't provide a valid ans**.

    Context:
    {context}

    Query: {query}

    Answer:
    """

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": llm_model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature
    }

    try:
        response = requests.post(API_URL, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            response_data = response.json()
            return response_data['choices'][0]['message']['content'].strip()
        else:
            return f"Error {response.status_code}: {response.text}"
    except requests.RequestException as e:
        return f"API request failed: {str(e)}"

""" Example Usage """
if __name__ == "__main__":
    query = "What is Django?"
    retrieved = retrieve_documents(query, top_k=2)
    print("Retrieved Documents:", retrieved)

    response = generate_response(query, retrieved)
    print("Generated Response:", response)
