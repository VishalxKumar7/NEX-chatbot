import os
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from pdf_processor import load_pdf
from splitter import split_documents

# Load environment variables from .env file
load_dotenv()  # automatically reads .env in project root

def create_chroma_embeddings(
    data_dir="backend/data", 
    collection_name="pdf_embeddings",
    hf_model_name="sentence-transformers/all-MiniLM-L6-v2"
):
    """
    1. Load PDF files
    2. Split them into chunks
    3. Generate embeddings using Hugging Face (from .env API key)
    4. Store embeddings in ChromaDB
    """

    # -----------------------------
    # STEP 1: Load and split PDFs
    # -----------------------------
    documents = load_pdf(data_dir)
    chunks = split_documents(documents)

    # -----------------------------
    # STEP 2: Load Hugging Face embedding model
    # -----------------------------
    hf_token = os.getenv("HUGGINGFACE_API_KEY")  # read from .env
    if not hf_token:
        raise ValueError("❌ HUGGINGFACEHUB_API_TOKEN not found in .env file!")

    print("⏳ Loading Hugging Face model...")
    model = SentenceTransformer(hf_model_name, use_auth_token=hf_token)
    print("✅ Hugging Face embedding model loaded successfully.")

    # -----------------------------
    # STEP 3: Initialize Chroma
    # -----------------------------
    client = chromadb.Client()

    if collection_name in [c.name for c in client.list_collections()]:
        client.delete_collection(collection_name)
        print(f"🧹 Old collection '{collection_name}' deleted.")

    collection = client.create_collection(
        name=collection_name,
        metadata={"source": "pdf"}
    )
    print(f"✅ Created new Chroma collection: {collection_name}")

    # -----------------------------
    # STEP 4: Add embeddings
    # -----------------------------
    for i, doc in enumerate(chunks):
        embedding = model.encode(doc.page_content)
        collection.add(
            documents=[doc.page_content],
            metadatas=[{"source": doc.metadata.get("source", "unknown")}],
            ids=[str(i)],
            embeddings=[embedding]
        )

    print(f"✅ Added {len(chunks)} chunks to Chroma collection '{collection_name}' successfully.")
    return collection


if __name__ == "__main__":
    create_chroma_embeddings()
    print("🎉 Chroma vectorstore created and embeddings stored successfully!")
