import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pymupdf
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load the embedding model into memory
model = SentenceTransformer('all-MiniLM-L6-v2')

# Connect to the persistent ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("rag-docs")

# Set up the text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200,
    separators=['\n\n', '\n', ' ', '']
)

def extract_text_from_pdf(pdf_path):
    """Reads a PDF and returns all text as a single string."""
    doc = pymupdf.open(pdf_path)
    return "\n".join([page.get_text() for page in doc])

def ingest_pdf(pdf_path):
    """
    Extracts, chunks, embeds, and stores a PDF in ChromaDB.
    Returns the number of chunks processed.
    """
    # Standardize the path and extract the filename for metadata
    clean_path = pdf_path.replace('\\', '/')
    source_filename = os.path.basename(clean_path)
    
    # Extract and chunk
    raw_text = extract_text_from_pdf(pdf_path)
    chunks = splitter.split_text(raw_text)
    
    # If the PDF was empty or unreadable, exit early
    if not chunks:
        return 0
        
    # Generate embeddings (progress bar disabled to keep UI logs clean)
    embeddings = model.encode(chunks, show_progress_bar=False)
    
    # Generate unique IDs for the chunks
    # We prefix with the filename so chunks from different PDFs don't overwrite each other
    safe_name = source_filename.replace(" ", "_")
    ids = [f"{safe_name}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": source_filename, "chunk": i} for i in range(len(chunks))]
    
    # Add to ChromaDB
    collection.add(
        documents=chunks,
        embeddings=embeddings.tolist(),
        ids=ids,
        metadatas=metadatas
    )
    
    return len(chunks)
def get_all_chunks():
    """
    Retrieves all document chunks from the ChromaDB collection.
    Returns a list of strings (the document chunks).
    """
    results = collection.get()
    if not results["documents"] :
        return []
        
    return results["documents"]

def retrieve(query, n=3):
    """
    Finds the 'n' most relevant document chunks for a given query.
    """

    q_embed = model.encode([query]).tolist()
    results = collection.query(query_embeddings=q_embed, n_results=n)
    if not results["documents"] or not results["documents"][0]:
        return []
        
    return results["documents"][0]