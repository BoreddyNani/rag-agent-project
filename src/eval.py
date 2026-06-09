import json, time, os, warnings
from pathlib import Path
from datasets import Dataset

# Hide noisy third-party FutureWarnings and DeprecationWarnings from terminal
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from ragas import evaluate
# FIX: Updated import path for Ragas metrics to stop deprecation warnings
from ragas.metrics import faithfulness, context_recall, answer_relevancy

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
os.environ["GEMINI_API_KEY"] = ""  # replace with your Gemini API key

# 1. Point RAGAS at Gemini as the judge LLM
gemini = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", # or gemini-1.5-flash
    google_api_key=os.environ["GEMINI_API_KEY"]
)
ragas_llm = LangchainLLMWrapper(gemini)

# 2. Initialize and wrap Google's embedding model
gemini_embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2",
    google_api_key=os.environ["GEMINI_API_KEY"]
)
ragas_embeddings = LangchainEmbeddingsWrapper(gemini_embeddings)

DATA_DIR = ROOT_DIR / "data"

# Load your QA pairs
with open(DATA_DIR / "eval_questions.json") as f:
    qa_pairs = json.load(f)

from rag_chain import rag_query
from retrieval import hybrid_retrieve

rows = []
print("Building dataset and querying RAG chain...")

for item in qa_pairs:
    # FIX: Robust retry logic for 503 Server Unavailable or 429 Rate Limits
    max_retries = 3
    for attempt in range(max_retries):
        try:
            result = rag_query(item["question"])
            contexts = hybrid_retrieve(item["question"], n=5)
            
            rows.append({
                "question":     item["question"],
                "answer":       result["answer"],
                "contexts":     contexts,
                "ground_truth": item["ground_truth"]
            })
            print(f"Processed: {item['question'][:30]}...")
            time.sleep(1) # Small delay to avoid hitting rate limits
            break # Success, break out of retry loop
            
        except Exception as e:
            if "503" in str(e) or "429" in str(e):
                print(f"Google API busy (Attempt {attempt + 1}/{max_retries}). Retrying in 10 seconds...")
                time.sleep(10)
                if attempt == max_retries - 1:
                    print("Max retries reached. Crashing gracefully.")
                    raise e
            else:
                raise e # If it's a different error, raise it immediately

dataset = Dataset.from_list(rows)

print("\nStarting Ragas Evaluation...")
# Run evaluation
scores = evaluate(
    dataset,
    metrics=[faithfulness, context_recall, answer_relevancy],
    llm=ragas_llm,
    embeddings=ragas_embeddings
)

print("\nEvaluation Complete!")
print(scores)

# Save results
with open(DATA_DIR / "ragas_results_week2.json", "w") as f:
    json.dump(scores.to_pandas().to_dict(), f, indent=2)