from langchain_text_splitters import RecursiveCharacterTextSplitter
from pdf_processor import load_pdf

docs =  load_pdf("backend/data")

def split_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=60,
        separators=["\n\n", "\n", " ", ""]
    )
    split_docs = text_splitter.split_documents(docs)
    return split_docs

split = split_documents(docs)
print(f"Split into {len(split)} documents after text splitting.")
