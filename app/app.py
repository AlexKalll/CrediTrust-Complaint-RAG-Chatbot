import gradio as gr
import pandas as pd
from pathlib import Path
from datetime import datetime
from src.rag import RAGPipeline
from transformers import TextIteratorStreamer
from threading import Thread
import csv
import os

# Initialize RAG pipeline
rag = RAGPipeline(
    index_path=Path("vectorstore/faiss_index.bin"),
    metadata_path=Path("vectorstore/metadata.parquet")
)

# Get unique products for dropdown
metadata = pd.read_parquet("vectorstore/metadata.parquet")
PRODUCTS = ["All Products"] + sorted(metadata['product'].unique().tolist())

# Conversation history storage
conversation_history = []
os.makedirs("conversation_logs", exist_ok=True)

def save_conversation():
    """Save conversation history to CSV"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"conversation_logs/conversation_{timestamp}.csv"
    
    with open(filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["timestamp", "question", "answer", "product_filter", "feedback"])
        writer.writeheader()
        writer.writerows(conversation_history)
    return filename

def respond(question: str, product_filter: str, history: list) -> tuple:
    """Handle user question with streaming"""
    # Apply product filter
    filter_metadata = metadata
    if product_filter != "All Products":
        filter_metadata = metadata[metadata['product'] == product_filter]
    
    # Get retrieved chunks
    query_embed = rag.retriever.embed_query(question)
    chunks, sources = rag.retriever.retrieve_chunks(query_embed)
    
    # Prepare streaming
    prompt = rag.generator._build_prompt(question, chunks)
    streamer = TextIteratorStreamer(rag.generator.pipe.tokenizer)
    
    # Start generation
    generation_kwargs = dict(
        prompt,
        streamer=streamer,
        max_new_tokens=200,
        temperature=0.3
    )
    thread = Thread(target=rag.generator.pipe, kwargs=generation_kwargs)
    thread.start()
    
    # Stream response
    full_response = ""
    for new_text in streamer:
        full_response += new_text
        if "Answer:" in full_response:
            answer_part = full_response.split("Answer:")[1].strip()
            yield answer_part, format_sources(sources), gr.update(interactive=True), gr.update(interactive=True)
    
    thread.join()
    
    # Log conversation
    conversation_history.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "answer": full_response.split("Answer:")[1].strip(),
        "product_filter": product_filter,
        "feedback": None
    })

def format_sources(sources: list) -> str:
    """Format sources for display"""
    return "\n\n".join(
        f"📌 **{src['product']}** (relevance: {src['score']:.2f}):\n"
        f"> {src['excerpt'][:150]}..."
        for src in sources[:2]
    )

def record_feedback(feedback: str):
    """Record user feedback"""
    if conversation_history:
        conversation_history[-1]["feedback"] = feedback
    return "Thank you for your feedback!"

with gr.Blocks(
    title="CreditTrust AI Analyst",
    theme=gr.themes.Soft(),
    css="""
    .gradio-container {max-width: 800px !important}
    .feedback-btns {margin-top: 10px}
    """
) as demo:
    # Header
    gr.Markdown("""
    <div style='text-align: center'>
        <h1>💳 CreditTrust Complaint Analyst</h1>
        <p>Ask natural language questions about customer feedback</p>
    </div>
    """)
    
    # Controls
    with gr.Row():
        product_dropdown = gr.Dropdown(
            label="Filter by Product",
            choices=PRODUCTS,
            value="All Products"
        )
        clear_btn = gr.Button("🧹 Clear Chat", size="sm")
        export_btn = gr.Button("💾 Export Conversation", size="sm")
    
    # Chat interface
    chatbot = gr.Chatbot(height=400, avatar_images=("assets/user.png", "assets/bot.png"))
    question = gr.Textbox(label="Ask a question", placeholder="E.g. What are common issues with BNPL services?")
    
    # Feedback buttons
    with gr.Row(visible=False) as feedback_row:
        thumbs_up = gr.Button("👍 Helpful", variant="primary", size="sm")
        thumbs_down = gr.Button("👎 Not Helpful", variant="secondary", size="sm")
    feedback_msg = gr.Textbox(visible=False)
    
    # Hidden components
    response = gr.Textbox(visible=False)
    sources = gr.Textbox(visible=False)
    
    # Interaction flow
    question.submit(
        fn=respond,
        inputs=[question, product_dropdown, chatbot],
        outputs=[response, sources, thumbs_up, thumbs_down]
    ).then(
        lambda r, s: ((None, r), (None, s)),
        inputs=[response, sources],
        outputs=[chatbot, chatbot]
    ).then(
        lambda: gr.update(visible=True),
        outputs=feedback_row
    )
    
    # Button actions
    clear_btn.click(
        lambda: ([], [], "", gr.update(visible=False)),
        outputs=[chatbot, question, feedback_msg, feedback_row]
    )
    
    export_btn.click(
        fn=save_conversation,
        outputs=gr.File(label="Download Conversation History")
    )
    
    thumbs_up.click(
        fn=lambda: record_feedback("positive"),
        outputs=feedback_msg
    ).then(
        lambda: gr.update(visible=False),
        outputs=feedback_row
    )
    
    thumbs_down.click(
        fn=lambda: record_feedback("negative"),
        outputs=feedback_msg
    ).then(
        lambda: gr.update(visible=False),
        outputs=feedback_row
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        favicon_path="assets/favicon.ico"
    )