import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys
import os

# Add the parent directory to the path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from src.rag_pipeline import RAGPipeline

# Page configuration
st.set_page_config(
    page_title="CreditTrust AI Analyst",
    page_icon="🏦",
    layout="wide"
)

# Initialize session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Initialize RAG pipeline
@st.cache_resource
def load_rag_pipeline():
    try:
        rag = RAGPipeline(
            index_path=root_dir / "vectorstore" / "faiss_index.bin",
            metadata_path=root_dir / "vectorstore" / "metadata.parquet"
        )
        return rag
    except Exception as e:
        st.error(f"Error loading RAG pipeline: {e}")
        return None

# Load products for filtering
@st.cache_data
def load_products():
    try:
        metadata = pd.read_parquet(root_dir / "vectorstore" / "metadata.parquet")
        return ["All Products"] + sorted(metadata['product'].unique().tolist())
    except Exception as e:
        st.error(f"Error loading products: {e}")
        return ["All Products"]

def main():
    st.title("🏦 CreditTrust AI Analyst")
    st.markdown("AI-powered complaint analysis and customer support system")
    
    # Load RAG pipeline
    rag = load_rag_pipeline()
    if rag is None:
        st.error("Failed to initialize RAG pipeline. Please check your setup.")
        return
    
    # Load products
    products = load_products()
    
    # Sidebar
    st.sidebar.header("Settings")
    product_filter = st.sidebar.selectbox(
        "Filter by Product",
        options=products,
        index=0
    )
    
    # Main chat interface
    st.header("💬 Chat with AI Analyst")
    
    # Chat input
    user_question = st.text_input(
        "Ask a question about complaints or financial products:",
        placeholder="e.g., What are common credit card billing issues?",
        key="user_input"
    )
    
    # Chat button
    if st.button("Ask Question", type="primary"):
        if user_question.strip():
            with st.spinner("Analyzing your question..."):
                try:
                    # Query RAG pipeline
                    result = rag.query(user_question, product_filter)
                    
                    # Display response
                    st.markdown("### 🤖 AI Response")
                    st.write(result['response'])
                    
                    # Display sources
                    if result['sources']:
                        st.markdown("### 📚 Sources")
                        for i, source in enumerate(result['sources'][:3]):
                            with st.expander(f"Source {i+1}: {source['product']} (Score: {source['score']:.2f})"):
                                st.write(f"**Issue:** {source['issue']}")
                                st.write(f"**Text:** {source['text'][:300]}...")
                    
                    # Add to conversation history
                    st.session_state.conversation_history.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "question": user_question,
                        "answer": result['response'],
                        "product_filter": product_filter,
                        "sources": result['sources']
                    })
                    
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    st.info("Please try again or check your setup.")
    
    # Display conversation history
    if st.session_state.conversation_history:
        st.header("📝 Conversation History")
        
        for i, conv in enumerate(reversed(st.session_state.conversation_history)):
            with st.expander(f"Q: {conv['question'][:50]}... ({conv['timestamp']})"):
                st.write(f"**Question:** {conv['question']}")
                st.write(f"**Answer:** {conv['answer']}")
                st.write(f"**Product Filter:** {conv['product_filter']}")
                
                if conv['sources']:
                    st.write("**Sources:**")
                    for j, source in enumerate(conv['sources'][:2]):
                        st.write(f"- {source['product']} (Score: {source['score']:.2f})")
    
    # Download conversation history
    if st.session_state.conversation_history:
        st.sidebar.header("Export")
        if st.sidebar.button("Download Conversation History"):
            df = pd.DataFrame(st.session_state.conversation_history)
            csv = df.to_csv(index=False)
            st.sidebar.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"conversation_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    # Clear conversation
    if st.sidebar.button("Clear Conversation"):
        st.session_state.conversation_history = []
        st.rerun()

if __name__ == "__main__":
    main()
