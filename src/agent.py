from pathlib import Path
from typing import TypedDict, List
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from src.retrieval import hybrid_retrieve
from youdotcom import You
from langchain_youdotcom import YouSearchTool


import re, os

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent

load_dotenv(ROOT_DIR/ ".env")

api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if not api_key:
    raise RuntimeError("Missing Gemini API key. Add GEMINI_API_KEY to src/.env")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",
                             google_api_key=api_key,
                             temperature=0)
you_key=os.environ.get("YOU_KEY")
you=You(api_key_auth=you_key)

class AgentState(TypedDict):
    query: str
    query_type: str
    retrieved_chunks: List[str]
    answer: str
    steps: List[str]
    chat_history: List[dict]

# ── Node 1: classify ────────────────────────────────────────────
def classify_query(state: AgentState) -> AgentState:
    history_ctx = ""
    if state["chat_history"]:
        recent = state["chat_history"][-4:]
        history_ctx = "Previous conversation:\n" + "\n".join(
            f"{m['role'].title()}: {m['content']}" for m in recent
        ) + "\n\n"

    prompt = f"""{history_ctx}Classify this query into exactly one of: factual, web, calculation.
- factual: can be answered from a document
- web: requires current/live information
- calculation: requires arithmetic or math

Query: {state['query']}
Return ONLY one word: factual, web, or calculation."""
    result = llm.invoke(prompt).content.strip().lower()
    query_type = result if result in ["factual","web","calculation"] else "factual"
    return {**state, "query_type": query_type,
            "steps": state["steps"] + [f"Classified as: {query_type}"]}

# ── Node 2a: retrieve and answer ────────────────────────────────
def retrieve_and_answer(state: AgentState) -> AgentState:
    chunks = hybrid_retrieve(state["query"], n=5)
    context = "\n\n".join(chunks)
    history_ctx = ""
    if state["chat_history"]:
        recent = state["chat_history"][-4:]
        history_ctx = "Conversation so far:\n" + "\n".join(
            f"{m['role'].title()}: {m['content']}" for m in recent
        ) + "\n\n"
    prompt = f"""Answer using ONLY this context. If not found, say so.
Context: {context}
Question: {state['query']}"""
    answer = llm.invoke(prompt).content
    return {**state, "retrieved_chunks": chunks, "answer": answer,
            "steps": state["steps"] + [f"Retrieved {len(chunks)} chunks"]}

# ── Node 2b: web search stub ─────────────────────────────────────
def web_search_and_answer(state: AgentState) -> AgentState:
    current_steps = state.get("steps", [])
    query=state["query"]
    snippets = []
    step_log = "Web search initialized"

    try:
        with You(api_key_auth=you_key) as you_c:

            response=you_c.search.unified(query=query,count=5)
            snippets=[]
            if response.results and response.results.web:
                for web_result in response.results.web:
                # Individual results store text in a list called snippets
                    if web_result.snippets:
                        snippets.extend(web_result.snippets)
                    
        # Join extracted snippets together
            context = "\n\n".join(snippets) if snippets else "No live snippets found."
            prompt=f""" using the following web search results, answer the question. Search results:{context} Question:{state['query']} Answer:"""
            answer= llm.invoke(prompt).content
            step_log = f"Successfully queried You.com and extracted {len(snippets)} snippets"
    except Exception as e:
        answer= f"{str(e)}"

    return {
        **state,
        "answer": answer,
        "retrieved_chunks": snippets[:5],
        "steps": current_steps + [step_log]
    }

# ── Node 2c: calculate ────────────────────────────────────────────
def calculate(state: AgentState) -> AgentState:
    current_steps = state.get("steps", [])
    math_prompt = f"""You are a math expression exporter. Convert the following text query into a raw mathematical expression.
                        Only include digits, operators (+, -, *, /, **), decimals, and parentheses. Do not include spaces or words.
                        Query: {state["query"]}
                        Expression:"""
    try:
        expression = llm.invoke(math_prompt).content.strip()
        # Validate the expression to contain only allowed characters
        if not re.match(r'^[\d+\-*/().\s]+$', expression):
            raise ValueError("Invalid characters in expression")
        # Evaluate the expression safely
        answer = str(eval(expression))
    except Exception as e:
        answer = f"Error occurred while evaluating the expression: {e}"

    return {**state, "answer": answer,
            "steps": current_steps + [f"Calculated: {state['query']}"]}
def summarise_if_long(state: AgentState) -> AgentState:
    if len(state["chat_history"]) <= 8:
        return state                    # not needed yet
    history_text = "\n".join(
        f"{m['role'].title()}: {m['content']}" for m in state["chat_history"]
    )
    summary = llm.invoke(
        f"Summarise this conversation in 3 sentences:\n{history_text}"
    ).content
    compressed = [{"role": "system", "content": f"Conversation summary: {summary}"}]
    return {**state, "chat_history": compressed,
            "steps": state["steps"] + ["Compressed history to summary"]}

# ── Route function ────────────────────────────────────────────────
def route(state: AgentState) -> str:
    return state["query_type"]   # "factual" | "web" | "calculation"

# ── Build graph ───────────────────────────────────────────────────
graph = StateGraph(AgentState)
graph.add_node("classify_query",        classify_query)
graph.add_node("retrieve_and_answer",   retrieve_and_answer)
graph.add_node("web_search_and_answer", web_search_and_answer)
graph.add_node("calculate",             calculate)
graph.add_node("summarise_if_long", summarise_if_long)

graph.add_edge(START, "classify_query")
graph.add_conditional_edges("classify_query", route, {
    "factual":     "retrieve_and_answer",
    "web":         "web_search_and_answer",
    "calculation": "calculate"
})

graph.add_edge("retrieve_and_answer",   "summarise_if_long")
graph.add_edge("web_search_and_answer", "summarise_if_long")
graph.add_edge("calculate",             "summarise_if_long")
graph.add_edge("summarise_if_long",     END)
agent = graph.compile()
