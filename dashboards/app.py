import gradio as gr
import pandas as pd
from pathlib import Path
from datetime import datetime
from transformers import TextIteratorStreamer
from threading import Thread
import csv
import os

import sys
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))
from src.rag_pipeline import RAGPipeline

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
    try:
        # Query RAG pipeline
        result = rag.query(question, product_filter)
        streamer = result['response_streamer']
        sources = result['sources']
        thread = result['generation_thread']
        thread.start()
        
        # Stream response
        full_response = ""
        for new_text in streamer:
            full_response += new_text
            # Yield the answer part if "Answer:" is present, otherwise yield the full response
            if "Answer:" in full_response:
                answer_part = full_response.split("Answer:")[1].strip()
                yield answer_part, format_sources(sources), gr.update(interactive=True), gr.update(interactive=True)
            else:
                # If "Answer:" is not yet in the response, yield the current full response
                yield full_response, format_sources(sources), gr.update(interactive=True), gr.update(interactive=True)
        
        thread.join()
        
        # After the thread joins, the full_response should contain the complete generated text.
        # Extract the answer part for logging, handling cases where "Answer:" might not be present.
        answer_to_log = full_response.split("Answer:")[1].strip() if "Answer:" in full_response else full_response.strip()
        
        # Log conversation
        conversation_history.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "question": question,
            "answer": answer_to_log,
            "product_filter": product_filter,
            "feedback": None
        })
        return answer_to_log, format_sources(sources), gr.update(interactive=True), gr.update(interactive=True)
    except Exception as e:
        error_message = f"An error occurred: {e}. Please try again."
        return error_message, "", gr.update(interactive=False), gr.update(interactive=False)

def format_sources(sources: list) -> str:
    """Format sources for display"""
    if not sources:
        return ""
    formatted_sources = []
    for i, src in enumerate(sources[:2]):
        formatted_sources.append(
            f"📌 **Source {i+1}: {src['product']}** (relevance: {src['score']:.2f}):\n"
            f"> {src['excerpt'][:150]}..."
        )
    return "\n\n".join(formatted_sources)

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
        lambda r, s: ((None, r), (None, s)) if r else (None, "An error occurred. Please try again."),
        inputs=[response, sources],
        outputs=[chatbot, chatbot]
    ).then(
        lambda r: gr.update(visible=True) if "An error occurred" not in r else gr.update(visible=False),
        inputs=[response],
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