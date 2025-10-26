import chromadb
from sentence_transformers import SentenceTransformer
import chromadb.errors

# -----------------------------
# CONFIG
# -----------------------------
PERSIST_DIRECTORY = "./backend/chroma_storage"
COLLECTION_NAME = "pdf_embeddings"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3  # number of chunks to retrieve

# -----------------------------
# STEP 1: Load persistent Chroma client
# -----------------------------
client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)

# -----------------------------
# STEP 2: Load saved collection
# -----------------------------
try:
    collection = client.get_collection(COLLECTION_NAME)
    print(f"✅ Collection '{COLLECTION_NAME}' loaded successfully!")
except chromadb.errors.NotFoundError:
    print(f"❌ Collection '{COLLECTION_NAME}' not found. Please create embeddings first!")
    exit()

# -----------------------------
# STEP 3: Load embedding model
# -----------------------------
model = SentenceTransformer(EMBEDDING_MODEL_NAME)
print("✅ Hugging Face embedding model loaded successfully.")

# -----------------------------
# STEP 4: Take user query
# -----------------------------
user_query = input("\nEnter your question: ")

# -----------------------------
# STEP 5: Convert query to embedding
# -----------------------------
query_embedding = model.encode(user_query)

# -----------------------------
# STEP 6: Query Chroma vectorstore
# -----------------------------
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=TOP_K
)

# -----------------------------
# STEP 7: Display results
# -----------------------------
print("\n🔍 Top Matching Results:")
print("=" * 80)

if results["documents"][0]:
    for i, doc in enumerate(results["documents"][0]):
        print(f"\nResult {i+1}:")
        print(doc.strip())
        print(f"Metadata: {results['metadatas'][0][i]}")
        print(f"Distance: {results['distances'][0][i]}")
        print("-" * 80)
else:
    print("No matching documents found for this query.")
