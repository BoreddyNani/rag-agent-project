import os
import pymupdf
import chromadb
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
pdf_path='C:\\Users\\tarun\\rag-agent-project\\data\\usf.pdf'
source_filename = os.path.basename(pdf_path)
def extract_text_from_pdf(pdf_path):
    doc=pymupdf.open(pdf_path)
    return "\n".join([page.get_text() for page in doc])


splitter=RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    separators=['\n\n', '\n', ' ', ''])

chunks=splitter.split_text(extract_text_from_pdf(pdf_path))

model=SentenceTransformer('all-MiniLM-L6-v2')
embeddings=model.encode(chunks, show_progress_bar=True)
print(f'Embeddings shape: {embeddings.shape}')

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("rag-docs")
collection.add(
    documents=chunks,
    embeddings=embeddings.tolist(),
    ids=[f"chunk_{i}" for i in range(len(chunks))],
    metadatas=[{"source": source_filename, "chunk": i} for i in range(len(chunks))]
)


def retrieve(query, n=3):
    q_embed = model.encode([query]).tolist()
    results = collection.query(query_embeddings=q_embed, n_results=n)
    return results["documents"][0]

print(retrieve("what are different types of Master’s degrees offered at USF?"))