from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "data")

print("Looking for PDFs in:", data_dir)
print("Files in folder:", os.listdir(data_dir))

def load_pdf(data_dir):
    loader = DirectoryLoader(
        data_dir,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents

documents = load_pdf(data_dir)
print(f"Loaded {len(documents)} documents from PDF files.")
