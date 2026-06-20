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
    # Call your RAG chain
    chat_hist = []
    for item in history:
        chat_hist.append(item)
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

    
    return answer 

# Build the UI
with gr.Blocks(title="RAG Document Chat") as demo:
    gr.Markdown("## Chat with your documents")

    with gr.Row():
        upload = gr.File(label="Upload PDF", file_types=[".pdf"])
        status = gr.Textbox(label="Status", interactive=False)

    # Cleaned up upload event - it only updates the status box now
    upload.upload(fn=upload_pdf, inputs=[upload], outputs=[status])

    chatbot = gr.ChatInterface(
        fn=chat,
        chatbot=gr.Chatbot(height=400),
        textbox=gr.Textbox(placeholder="Ask a question about your document..."),
        examples=["Summarize the main points", "What are the key requirements?"],
        cache_examples=False
    )

if __name__ == "__main__":
    demo.launch()