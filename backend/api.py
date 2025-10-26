import os
from fastapi import FastAPI
from pydantic import BaseModel
import chromadb
import chromadb.errors
from sentence_transformers import SentenceTransformer

# ----------------------------
# CONFIG
# ----------------------------
CURRENT_DIR = os.path.dirname(__file__)
PERSIST_DIRECTORY = os.path.join(CURRENT_DIR, "chroma_storage")
COLLECTION_NAME = "pdf_embeddings"
HF_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3

# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI(title="PDF RAG Chatbot API")

# ----------------------------
# Load Chroma collection
# ----------------------------
client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
try:
    collection = client.get_collection(COLLECTION_NAME)
    print(f"✅ Collection '{COLLECTION_NAME}' loaded successfully!")
except chromadb.errors.NotFoundError:
    print(f"❌ Collection '{COLLECTION_NAME}' not found. Run your embedding/vectorstore script first!")
    exit()

# ----------------------------
# Load embedding model
# ----------------------------
model = SentenceTransformer(HF_MODEL_NAME)
print("✅ Embedding model loaded successfully")

# ----------------------------
# Request model
# ----------------------------
class Query(BaseModel):
    chat_id: str
    question: str

# In-memory chat history
chat_history = {}

@app.post("/ask")
def ask(query: Query):
    # Debug: show received payload
    print("Received:", query.dict())

    # Validate non-empty question
    if not query.question.strip():
        return {"answer": "Please provide a question.", "history": chat_history.get(query.chat_id, [])}

    # Convert query to embedding
    query_embedding = model.encode(query.question)

    # Query Chroma
    results = collection.query(query_embeddings=[query_embedding], n_results=TOP_K)
    top_doc = results["documents"][0][0] if results["documents"][0] else "No matching document found"

    # Update chat history
    if query.chat_id not in chat_history:
        chat_history[query.chat_id] = []
    chat_history[query.chat_id].append({"question": query.question, "answer": top_doc})

    return {"answer": top_doc, "history": chat_history[query.chat_id]}
