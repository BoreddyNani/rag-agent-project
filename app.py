import gradio as gr
from src.ingestion import ingest_pdf, retrieve
from src.rag_chain import rag_query
from src.retrieval import hybrid_retrieve, initialize_bm25_index
from src.agent import agent

def upload_pdf(file):
    if file is None:
        return "No file uploaded."
    
    try:
        # file.name contains the temporary path where Gradio stored the upload
        chunk_count = ingest_pdf(file.name)
        # Handle Windows/Linux path splitting gracefully
        initialize_bm25_index()
        filename = file.name.replace('\\', '/').split('/')[-1]
        return f"Success! Ingested {chunk_count} chunks from {filename}"
    except Exception as e:
        return f"Error during ingestion: {str(e)}"

def chat(message, history):
    # Convert Gradio chat history to list of dicts for agent
    chat_hist = [item for item in history]
    inputs = {
            "query": message,
            "query_type": "",
            "retrieved_chunks": [],
            "answer": "",
            "steps": [],
            "chat_history": chat_hist
        }
    result = agent.invoke(inputs)
    answer = result["answer"]
    chunks_list = result.get("retrieved_chunks", [])
    steps_list = result.get("steps", [])
    
    # Format sources
    sources_md = ""
    if chunks_list:
        sources_md = "\n\n---\n\n".join(f"📄 **Chunk [{i+1}]:**\n{chunk[:500]}..." for i, chunk in enumerate(chunks_list[:5]))
    else:
        sources_md = "*No chunks retrieved.*"
    
    # Format steps
    steps_md = ""
    if steps_list:
        steps_md = "\n".join(f"🔄 {step}" for step in steps_list)
    else:
        steps_md = "*No steps recorded.*"
    
    return answer, sources_md, steps_md

# Build the UI
with gr.Blocks(title="RAG Document Chat") as demo:
    gr.Markdown("## Chat with your documents")

    with gr.Row():
        with gr.Column(scale=4):
            with gr.Row():
                upload = gr.File(label="Upload PDF", file_types=[".pdf"])
                status = gr.Textbox(label="Status", interactive=False)

            upload.upload(fn=upload_pdf, inputs=[upload], outputs=[status])

            chatbot = gr.Chatbot(height=400)
            msg = gr.Textbox(placeholder="Ask a question about your document...", show_label=False)
            
        with gr.Column(scale=3):
            gr.Markdown("### 🛠️ Agent Internals Tracker")
            
            with gr.Accordion("🔍 Agent Decision Process / Steps", open=True):
                reasoning_panel = gr.Markdown("*Awaiting user query...*")
                
            with gr.Accordion("📚 Retrieved Grounding Context Sources", open=True):
                sources_panel = gr.Markdown("*No context loaded yet.*")
    
    def respond(message, history):
        answer, sources, steps = chat(message, history)
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": answer})
        return history, sources, steps, ""
    
    msg.submit(respond, [msg, chatbot], [chatbot, sources_panel, reasoning_panel, msg])

if __name__ == "__main__":
    demo.launch()