import json
import sys
import time
from pathlib import Path

from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.append(str(SCRIPT_DIR))

from ingestion import retrieve  
from retrieval import hybrid_retrieve  

# Initialize Gemini (gemini-1.5-flash is the equivalent to Haiku for speed/cost)

ai_client = genai.Client(api_key="")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key="",
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

DATA_DIR = ROOT_DIR / "data"

# Run all test queries
with open(DATA_DIR / "eval_questions.json") as f:
    questions = json.load(f)

results = [rag_query(q["question"]) for q in questions]

with open(DATA_DIR / "test_results_hybrid.json", "w") as f:
    json.dump(results, f, indent=2)

# Print latency summary
latencies = [r["latency_ms"] for r in results]
print(f"Avg latency: {sum(latencies)//len(latencies)}ms")
print(f"Max latency: {max(latencies)}ms")