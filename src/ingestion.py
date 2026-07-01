import os
import pymupdf
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Disable oneDNN optimizations to keep startup logs clean
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load the embedding model into memory
model = SentenceTransformer('all-MiniLM-L6-v2')

# ── DYNAMIC ENVIRONMENT DETECTION FOR CHROMA ───────────────────
chroma_host = os.environ.get("CHROMA_HOST", "").strip()

if chroma_host == "chromadb":
    # Docker Compose path: Talk to your dedicated container service
    print("[CHROMA] Docker architecture detected. Connecting via HttpClient...")
    client = chromadb.HttpClient(host="chromadb", port=8000)
else:
    # Hugging Face / Render path: Store files directly to local workspace memory
    # Hugging Face Spaces give you write access to your root app directory out-of-the-box
    persist_dir = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")
    print(f"[CHROMA] Single container environment detected. Using PersistentClient at: {persist_dir}")
    client = chromadb.PersistentClient(path=persist_dir)

collection = client.get_or_create_collection("rag-docs")
# ───────────────────────────────────────────────────────────────

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
    """Extracts, chunks, embeds, and stores a PDF in ChromaDB."""
    clean_path = pdf_path.replace('\\', '/')
    source_filename = os.path.basename(clean_path)
    
    raw_text = extract_text_from_pdf(pdf_path)
    chunks = splitter.split_text(raw_text)
    
    if not chunks:
        return 0
        
    embeddings = model.encode(chunks, show_progress_bar=False)
    
    safe_name = source_filename.replace(" ", "_")
    ids = [f"{safe_name}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": source_filename, "chunk": i} for i in range(len(chunks))]
    
    collection.add(
        documents=chunks,
        embeddings=embeddings.tolist(),
        ids=ids,
        metadatas=metadatas
    )
    return len(chunks)

def get_all_chunks():
    """Retrieves all document chunks from the ChromaDB collection."""
    results = collection.get()
    if not results["documents"]:
        return []
    return results["documents"]

def retrieve(query, n=3):
    """Finds the 'n' most relevant document chunks for a given query."""
    q_embed = model.encode([query]).tolist()
    results = collection.query(query_embeddings=q_embed, n_results=n)
    if not results["documents"] or not results["documents"][0]:
        return []
    return results["documents"][0]