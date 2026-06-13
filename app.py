import gradio as gr
from ingestion import ingest_pdf, retrieve
from rag_chain import rag_query
from retrieval import hybrid_retrieve
def upload_pdf(file):
    if file is None:
        return "No file uploaded."
    
    try:
        # file.name contains the temporary path where Gradio stored the upload
        chunk_count = ingest_pdf(file.name)
        # Handle Windows/Linux path splitting gracefully
        filename = file.name.replace('\\', '/').split('/')[-1]
        return f"Success! Ingested {chunk_count} chunks from {filename}"
    except Exception as e:
        return f"Error during ingestion: {str(e)}"

def chat(message, history):
    # Call your RAG chain
    result = rag_query(message)
    answer = result["answer"]

    # Fetch sources (Note: It is more efficient to have rag_query return 
    # the chunks directly so you don't have to call retrieve() twice)
    sources = hybrid_retrieve(message, n=8)
    source_text = "\n\n**Sources used:**\n" + "\n".join(
        f"- {s[:150]}..." for s in sources
    )
    
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
        examples=["Summarize the main points", "What are the key requirements?"]
    )

if __name__ == "__main__":
    demo.launch()