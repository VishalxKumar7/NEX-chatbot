import os
import chromadb
from sentence_transformers import SentenceTransformer
from pdf_processor import load_pdf
from splitter import split_documents

# Persistent storage directory
PERSIST_DIRECTORY = "./backend/chroma_storage"

def create_vectorstore(
    data_dir="backend/data",
    collection_name="pdf_embeddings",
    hf_model_name="sentence-transformers/all-MiniLM-L6-v2"
):
    documents = load_pdf(data_dir)
    split_docs = split_documents(documents)
    print(f"Split into {len(split_docs)} chunks.")

    model = SentenceTransformer(hf_model_name)
    print("✅ Hugging Face embedding model loaded successfully.")

    # Initialize persistent Chroma client
    client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)

    # Delete old collection if exists
    if collection_name in [c.name for c in client.list_collections()]:
        client.delete_collection(collection_name)
        print(f"🧹 Old collection '{collection_name}' deleted.")

    # Create new collection
    collection = client.create_collection(
        name=collection_name,
        metadata={"source": "pdf"}
    )
    print(f"✅ Created new Chroma collection: {collection_name}")

    # Add embeddings
    for i, doc in enumerate(split_docs):
        embedding = model.encode(doc.page_content)
        collection.add(
            documents=[doc.page_content],
            metadatas=[{"source": doc.metadata.get("source", "pdf")}],
            ids=[str(i)],
            embeddings=[embedding]
        )

    print(f"✅ Added {len(split_docs)} chunks to Chroma collection '{collection_name}' successfully.")
    return collection

if __name__ == "__main__":
    create_vectorstore()
    print("🎉 Persistent Chroma vectorstore created successfully!")
