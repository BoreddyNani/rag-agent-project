import json
import sys
import time
from pathlib import Path
import os

from dotenv import load_dotenv
from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.append(str(SCRIPT_DIR))
load_dotenv(ROOT_DIR/ ".env")

from ingestion import retrieve  
from retrieval import hybrid_retrieve  

# Initialize Gemini (gemini-1.5-flash is the equivalent to Haiku for speed/cost)
api_key_from_env = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if not api_key_from_env:
    raise RuntimeError("Missing Gemini API key. Add GEMINI_API_KEY to src/.env")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key_from_env,
    temperature=0.2)

PROMPT = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the question using ONLY
the context below. If the answer is not in the context,
say "I don't know based on the provided documents."

Context:
{context}

Question: {question}
Answer:""")

def rag_query(question: str) -> dict:
    start = time.time()

    # retrieve top-5 chunks
    chunks = hybrid_retrieve(question, n=5)
    context = "\n\n".join(chunks)

    # call LLM
    chain = PROMPT | llm
    answer = chain.invoke({"context": context, "question": question})

    latency_ms = round((time.time() - start) * 1000)
    return {
        "question": question,
        "answer": answer.content,
        "chunks_used": len(chunks),
        "latency_ms": latency_ms
    }

