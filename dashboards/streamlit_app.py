import streamlit as st
import pandas as pd
from pathlib import Path
import sys
import os
from datetime import datetime
import json
import base64
from io import BytesIO
import time

# Try to import plotly, but handle gracefully if not available
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("⚠️ Plotly not available. Charts will be displayed as simple text.")

# Add root directory to path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

# Import RAG pipeline with graceful fallback
try:
    from src.rag_pipeline import RAGPipeline
    from src.data_preprocessing import load_data
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    st.warning(f"⚠️ Some imports are not available: {e}")
    st.info("The app will run in demo mode with limited functionality.")
    
    # Create dummy classes for demo mode
    class DummyRAGPipeline:
        def query(self, prompt, product_filter):
            return {
                'response': 'This is a demo response. The full RAG pipeline is not available in this environment.',
                'sources': []
            }
    
    class DummyDataLoader:
        @staticmethod
        def load_data(file_path):
            return pd.DataFrame({
                'product': ['Credit Card', 'Personal Loan', 'Mortgage', 'Checking Account', 'Savings Account'],
                'Issue': ['Billing Error', 'Unclear Terms', 'High Interest', 'Fees', 'Customer Service'],
                'narrative': ['Sample complaint text 1', 'Sample complaint text 2', 'Sample complaint text 3', 'Sample complaint text 4', 'Sample complaint text 5']
            })
    
    RAGPipeline = DummyRAGPipeline
    load_data = DummyDataLoader.load_data

# Page configuration
st.set_page_config(
    page_title="CreditTrust AI Analyst",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        padding: 2.5rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="50" cy="50" r="1" fill="white" opacity="0.1"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>');
        opacity: 0.3;
    }
    
    .main-header h1 {
        font-size: 3rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        font-size: 1.3rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        transition: all 0.3s ease;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.2);
    }
    
    .metric-card h3 {
        font-size: 1.1rem;
        margin: 0 0 1rem 0;
        opacity: 0.9;
        font-weight: 600;
    }
    
    .metric-card h2 {
        font-size: 2.5rem;
        margin: 0;
        font-weight: 800;
    }
    
    /* Chat message styling */
    .chat-message {
        padding: 1.5rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        border-left: 6px solid;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .chat-message:hover {
        transform: translateX(5px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    
    .user-message {
        background: linear-gradient(135deg, #f0f2f6 0%, #e8eaed 100%);
        border-left-color: #667eea;
        margin-left: 2rem;
    }
    
    .bot-message {
        background: linear-gradient(135deg, #e8f4fd 0%, #d1ecf1 100%);
        border-left-color: #28a745;
        margin-right: 2rem;
    }
    
    /* Source cards */
    .source-card {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 4px solid #28a745;
        box-shadow: 0 3px 10px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
        color: #333;
        border: 1px solid #e9ecef;
    }
    
    .source-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.12);
        border-color: #28a745;
    }
    
    .source-card strong {
        color: #28a745;
    }
    
    .source-card em {
        color: #666;
    }
    
    .source-card p {
        color: #333;
        line-height: 1.5;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 30px;
        padding: 0.8rem 2.5rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div > div {
        background: white;
        border-radius: 15px;
        border: 2px solid #e9ecef;
        transition: all 0.3s ease;
        color: #333;
    }
    
    .stSelectbox > div > div > div:focus-within {
        border-color: #667eea;
        box-shadow: 0 0 0 0.2rem rgba(102, 126, 234, 0.25);
    }
    
    .stSelectbox > div > div > div > div {
        color: #333;
        background: white;
    }
    
    .stSelectbox > div > div > div > div > div {
        color: #333;
        background: white;
    }
    
    
    /* Chart containers */
    .chart-container {
        background: white;
        padding: 0.3rem;
        border-radius: 13px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        margin: 1rem 0;
    }
    
    /* Progress bars */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 10px;
        border: 1px solid #dee2e6;
        font-weight: 600;
        color: #333;
    }
    
    .streamlit-expanderContent {
        background: white;
        color: #333;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin-top: 0.5rem;
    }
    
    /* Alert styling */
    .stAlert {
        border-radius: 15px;
        border: none;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Footer styling */
    .footer {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin-top: 3rem;
        border: 1px solid #dee2e6;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        .main-header p {
            font-size: 1rem;
        }
        .metric-card {
            padding: 1.5rem;
        }
        .metric-card h2 {
            font-size: 2rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_product_filter' not in st.session_state:
    st.session_state.current_product_filter = "All Products"
if 'show_analytics' not in st.session_state:
    st.session_state.show_analytics = True
if 'show_sample_questions' not in st.session_state:
    st.session_state.show_sample_questions = False

# Initialize RAG pipeline
@st.cache_resource
def initialize_rag():
    try:
        rag = RAGPipeline(
            index_path=root_dir / "vectorstore" / "faiss_index.bin",
            metadata_path=root_dir / "vectorstore" / "metadata.parquet"
        )
        return rag
    except Exception as e:
        st.error(f"Failed to initialize RAG pipeline: {e}")
        return None

# Load data for analytics
@st.cache_data
def load_analytics_data():
    try:
        # Use the filtered complaints CSV file
        data_file = root_dir / "data" / "filtered_complaints.csv"
        if data_file.exists():
            data = load_data(str(data_file))
            
            # Handle column mapping for analytics
            column_mapping = {
                'Product': 'product',
                'Consumer complaint narrative': 'narrative',
                'Issue': 'Issue'  # Keep as is if exists
            }
            
            # Rename columns that exist
            for old_name, new_name in column_mapping.items():
                if old_name in data.columns:
                    data = data.rename(columns={old_name: new_name})
            
            # Ensure we have the required 'product' column for analytics
            if 'product' not in data.columns:
                st.warning("⚠️ 'Product' column not found in data. Analytics may be limited.")
            
            return data
        else:
            st.warning(f"Data file not found at {data_file}. Using demo data for demonstration.")
            # Create demo data for demonstration
            demo_data = pd.DataFrame({
                'product': ['Credit Card', 'Personal Loan', 'Mortgage', 'Checking Account', 'Savings Account'],
                'Issue': ['Billing Error', 'Unclear Terms', 'High Interest', 'Fees', 'Customer Service'],
                'narrative': ['Sample complaint text 1', 'Sample complaint text 2', 'Sample complaint text 3', 'Sample complaint text 4', 'Sample complaint text 5']
            })
            return demo_data
    except Exception as e:
        st.warning(f"Failed to load data: {e}. Using demo data for demonstration.")
        # Create demo data for demonstration
        demo_data = pd.DataFrame({
            'product': ['Credit Card', 'Personal Loan', 'Mortgage', 'Checking Account', 'Savings Account'],
            'Issue': ['Billing Error', 'Unclear Terms', 'High Interest', 'Fees', 'Customer Service'],
            'narrative': ['Sample complaint text 1', 'Sample complaint text 2', 'Sample complaint text 3', 'Sample complaint text 4', 'Sample complaint text 5']
        })
        return demo_data

# Initialize components
rag = initialize_rag()
data = load_analytics_data()

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style='text-align: left; margin-top:-50px'>
        <h2>Control Panel</h2>
        <p style='color: #666; font-size: 0.9rem;'>Customize your experience</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Product filter
    if data is not None and 'product' in data.columns:
        products = ["All Products"] + sorted(data['product'].unique().tolist())
        selected_product = st.selectbox(
            "🔍 Filter by Product",
            options=products,
            index=0,
            key="product_filter",
            help="Select a specific product to focus your analysis"
        )
        st.session_state.current_product_filter = selected_product
    elif data is not None:
        st.info("⚠️ Product column not available for filtering")
        st.session_state.current_product_filter = "All Products"
    
    # Model settings
    st.markdown("### ⚙️ AI Settings")
    temperature = st.slider("🎨 Response Creativity", 0.0, 1.0, 0.7, 0.1, 
                           help="Higher values make responses more creative, lower values make them more focused")
    max_tokens = st.slider("📏 Max Response Length", 100, 1000, 500, 50,
                           help="Maximum number of words in AI responses")

    # Analytics toggle
    st.markdown("### 📊 Dashboard")
    show_analytics = st.checkbox("Show Analytics Dashboard", value=True, key="analytics_toggle")
    # Auto-refresh for real-time updates
    if st.button("🔄 Refresh Dashboard", key="refresh_dashboard", use_container_width=True):
        st.rerun()
    st.session_state.show_analytics = show_analytics
    
    # Export options
    st.markdown("### 📤 Export Options")
    if st.button("💾 Export Chat History", key="export_chat", use_container_width=True):
        if st.session_state.chat_history:
            export_data = []
            for msg in st.session_state.chat_history:
                export_data.append({
                    'timestamp': msg['timestamp'],
                    'role': msg['role'],
                    'content': msg['content'],
                    'product_filter': msg.get('product_filter', 'All Products')
                })
            
            df = pd.DataFrame(export_data)
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="chat_history.csv">📥 Download CSV</a>'
            st.markdown(href, unsafe_allow_html=True)
            st.success("✅ Chat history exported successfully!")
    
    # Clear chat
    if st.button("🧹 Clear Chat History", key="clear_chat", use_container_width=True):
        st.session_state.chat_history = []
        st.success("✅ Chat history cleared!")
        st.rerun()
    
    # System info
    st.markdown("### ℹ️ System Info")
    st.info(f"**RAG Pipeline:** {'✅ Active' if rag else '❌ Inactive'}")
    st.info(f"**Data Loaded:** {'✅ Yes' if data is not None else '❌ No'}")
    st.info(f"**Plotly Charts:** {'✅ Available' if PLOTLY_AVAILABLE else '❌ Not Available'}")
    st.info(f"**Imports:** {'✅ All Available' if IMPORTS_AVAILABLE else '⚠️ Demo Mode'}")
    
    # Deployment status
    if not IMPORTS_AVAILABLE or not PLOTLY_AVAILABLE:
        st.warning("⚠️ **Deployment Mode:** Some features are limited")
        st.info("This is normal for cloud deployments. The app will work with reduced functionality.")
    
    # Debug information
    if st.checkbox("🔧 Show Debug Info", key="debug_toggle"):
        st.markdown("### 🔍 Debug Details")
        
        # Check RAG pipeline status
        if rag:
            try:
                # Check if vectorstore files exist
                index_path = root_dir / "vectorstore" / "faiss_index.bin"
                metadata_path = root_dir / "vectorstore" / "metadata.parquet"
                
                st.info(f"**Index File:** {'✅ Exists' if index_path.exists() else '❌ Missing'}")
                st.info(f"**Metadata File:** {'✅ Exists' if metadata_path.exists() else '❌ Missing'}")
                
                if index_path.exists() and metadata_path.exists():
                    st.success("✅ Vector store files are present")
                else:
                    st.warning("⚠️ Some vector store files are missing")
                    
            except Exception as e:
                st.error(f"**Debug Error:** {e}")
        else:
            st.error("❌ RAG pipeline failed to initialize")
    
    if rag and IMPORTS_AVAILABLE:
        st.success("🚀 Ready to chat!")
    elif not IMPORTS_AVAILABLE:
        st.info("🎭 Demo mode - limited functionality")
    else:
        st.error("❌ Chat functionality unavailable")

# Main header
st.markdown("""
<div class="main-header">
    <h1>💳 CreditTrust AI Complaint Analyst</h1>
    <p>Intelligent Analysis of Customer Feedback & Complaints</p>
    <p>Powered by Advanced RAG Technology & Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# Data status message
if data is not None:
    if 'product' in data.columns:
        # st.success(f"✅ Data loaded successfully! {len(data)} complaints from {len(data['product'].unique())} products")
        pass
    else:
        st.warning("⚠️ Demo data loaded. For full functionality, ensure 'filtered_complaints.csv' exists in the data/ directory")
        
        # Show instructions for adding data
        with st.expander("📋 How to add your own data"):
            st.markdown("""
            To use your own complaint data:
            1. **Place your CSV file** in the `data/` directory
            2. **Name it** `filtered_complaints.csv`
            3. **Ensure it has these columns:**
               - `Product` - Product categories
               - `Consumer complaint narrative` - Complaint text
               - `Issue` - Issue types (optional)
               - `Complaint ID` - Unique identifier
            4. **Restart the app** to load your data
            """)
else:
    st.error("❌ Failed to load data. Please check the data directory and file structure.")

# Analytics Dashboard
if st.session_state.show_analytics and data is not None:
    st.markdown("## 📊 Analytics Dashboard")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>📈 Total Complaints</h3>
            <h2>{}</h2>
        </div>
        """.format(len(data)), unsafe_allow_html=True)
    
    with col2:
        if 'product' in data.columns:
            product_count = len(data['product'].unique())
        else:
            product_count = "N/A"
        st.markdown("""
        <div class="metric-card">
            <h3>🏷️ Products</h3>
            <h2>{}</h2>
        </div>
        """.format(product_count), unsafe_allow_html=True)
    
    with col3:
        if 'Issue' in data.columns:
            issue_count = len(data['Issue'].unique())
        else:
            issue_count = "N/A"
        st.markdown("""
        <div class="metric-card">
            <h3>⚠️ Issue Types</h3>
            <h2>{}</h2>
        </div>
        """.format(issue_count), unsafe_allow_html=True)
    
    with col4:
        if 'product' in data.columns and selected_product != "All Products":
            filtered_data = data[data['product'] == selected_product]
            filtered_count = len(filtered_data)
        else:
            filtered_count = len(data)
        st.markdown("""
        <div class="metric-card">
            <h3>🔍 Filtered Results</h3>
            <h2>{}</h2>
        </div>
        """.format(filtered_count), unsafe_allow_html=True)
    
    # Charts
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    
    with col1:
        # Product distribution
        if 'product' in data.columns:
            if PLOTLY_AVAILABLE:
                product_counts = data['product'].value_counts().head(10)
                fig = px.pie(
                    values=product_counts.values,
                    names=product_counts.index,
                    title="📊 Top 10 Products by Complaint Volume",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(
                    title_font_size=16,
                    title_font_color="#333",
                    showlegend=True,
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Fallback to text display
                product_counts = data['product'].value_counts().head(10)
                st.markdown("### 📊 Top 10 Products by Complaint Volume")
                for i, (product, count) in enumerate(product_counts.items(), 1):
                    percentage = (count / len(data)) * 100
                    st.markdown(f"**{i}.** {product}: {count} complaints ({percentage:.1f}%)")
        else:
            st.info("📊 Product data not available for charting")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Issue distribution (if available)
        if 'Issue' in data.columns:
            if PLOTLY_AVAILABLE:
                issue_counts = data['Issue'].value_counts().head(10)
                fig = px.bar(
                    x=issue_counts.values,
                    y=issue_counts.index,
                    orientation='h',
                    title="📋 Top 10 Issue Types",
                    color=issue_counts.values,
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(
                    title_font_size=16,
                    title_font_color="#333",
                    xaxis_title="Number of Complaints",
                    yaxis_title="Issue Type",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Fallback to text display
                issue_counts = data['Issue'].value_counts().head(10)
                st.markdown("### 📋 Top 10 Issue Types")
                for i, (issue, count) in enumerate(issue_counts.items(), 1):
                    percentage = (count / len(data)) * 100
                    st.markdown(f"**{i}.** {issue}: {count} complaints ({percentage:.1f}%)")
        else:
            # Timeline chart
            if 'date' in data.columns:
                if PLOTLY_AVAILABLE:
                    data['date'] = pd.to_datetime(data['date'], errors='coerce')
                    monthly_counts = data.groupby(data['date'].dt.to_period('M')).size()
                    fig = px.line(
                        x=monthly_counts.index.astype(str),
                        y=monthly_counts.values,
                        title="📅 Complaints Over Time",
                        markers=True
                    )
                    fig.update_layout(
                        title_font_size=16,
                        title_font_color="#333",
                        xaxis_title="Month",
                        yaxis_title="Number of Complaints",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # Fallback to text display
                    st.markdown("### 📅 Complaints Over Time")
                    st.info("Timeline chart not available without plotly. Please check the data for temporal patterns manually.")
            else:
                st.info("📊 No date column available for timeline analysis")
        st.markdown('</div>', unsafe_allow_html=True)

# Chat Interface
st.markdown("## 💬 AI Chat Interface")

# Initialize chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display sources for bot messages
        if message["role"] == "assistant" and "sources" in message:
            with st.expander(f"📚 Sources ({len(message['sources'])} found)"):
                for j, source in enumerate(message['sources'][:3]):
                    st.markdown(f"""
                    <div class="source-card">
                        <strong>Source {j+1}: {source.get('product', 'Unknown Product')}</strong><br>
                        <em>Relevance: {source.get('score', 0):.3f}</em><br>
                        <p>{source.get('excerpt', source.get('text', 'No content available'))[:300]}...</p>
                    </div>
                    """, unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask a question about customer complaints..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        if rag is None:
            st.error("❌ RAG pipeline is not available. Please check the setup.")
            # Add error message to chat
            st.session_state.messages.append({
                "role": "assistant", 
                "content": "❌ RAG pipeline is not available. Please check the setup.",
                "sources": []
            })
        else:
            with st.spinner("🤖 AI is analyzing your question..."):
                try:
                    # Query RAG pipeline with better error handling
                    if st.session_state.current_product_filter != "All Products":
                        # Add product context to the query
                        enhanced_prompt = f"Regarding {st.session_state.current_product_filter}: {prompt}"
                    else:
                        enhanced_prompt = prompt
                    
                    result = rag.query(enhanced_prompt, st.session_state.current_product_filter)
                    
                    # Extract response and sources with better fallbacks
                    if result and isinstance(result, dict):
                        response = result.get('response', '')
                        sources = result.get('sources', [])
                        
                        # If no response, provide a helpful message
                        if not response or response.strip() == "":
                            response = "I understand your question about customer complaints. However, I couldn't generate a specific response at this time. Please try rephrasing your question or check if the RAG pipeline has been properly trained with relevant data."
                        
                        # Display response
                        message_placeholder.markdown(response)
                        
                        # Add bot response to chat
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response,
                            "sources": sources
                        })
                        
                        # Display sources if available
                        if sources and len(sources) > 0:
                            with st.expander(f"📚 Sources ({len(sources)} found)"):
                                for j, source in enumerate(sources[:3]):
                                    source_text = source.get('excerpt', source.get('text', 'No content available'))
                                    source_product = source.get('product', 'Unknown Product')
                                    source_score = source.get('score', 0)
                                    
                                    st.markdown(f"""
                                    <div class="source-card">
                                        <strong>Source {j+1}: {source_product}</strong><br>
                                        <em>Relevance: {source_score:.3f}</em><br>
                                        <p>{source_text[:300]}{'...' if len(source_text) > 300 else ''}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                        else:
                            st.info("📚 No specific sources found for this query.")
                    
                    else:
                        # Handle case where result is not a dict
                        response = "I received an unexpected response format from the AI system. Please try again or contact support if the issue persists."
                        message_placeholder.markdown(response)
                        
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response,
                            "sources": []
                        })
                    
                except Exception as e:
                    # Better error handling with specific error messages
                    error_msg = f"❌ An error occurred while processing your request: {str(e)}"
                    
                    # Provide more helpful error messages
                    if "index" in str(e).lower() or "faiss" in str(e).lower():
                        error_msg = "❌ Vector database error: The AI system's knowledge base may not be properly initialized. Please check if the FAISS index exists."
                    elif "model" in str(e).lower() or "llm" in str(e).lower():
                        error_msg = "❌ AI model error: The language model may not be properly loaded. Please check the model configuration."
                    elif "memory" in str(e).lower() or "out of memory" in str(e).lower():
                        error_msg = "❌ Memory error: The system is running low on memory. Please try a simpler query or restart the application."
                    else:
                        error_msg = f"❌ Unexpected error: {str(e)}. Please try again or contact support."
                    
                    # Try to provide a helpful fallback response based on the question
                    fallback_response = ""
                    if "common" in prompt.lower() and "issue" in prompt.lower():
                        fallback_response = "Based on typical financial services data, common issues often include billing disputes, unclear terms, high fees, and customer service problems. For specific analysis, the AI system needs to be properly configured."
                    elif "product" in prompt.lower() and "complaint" in prompt.lower():
                        fallback_response = "Product-related complaints typically vary by financial service type. Credit cards often have billing issues, loans may have unclear terms, and accounts might have fee-related problems. The AI system can provide detailed analysis when properly configured."
                    elif "trend" in prompt.lower() or "pattern" in prompt.lower():
                        fallback_response = "Complaint trends and patterns can reveal important insights about service quality and customer satisfaction. The AI system can analyze temporal patterns and product-specific trends when fully operational."
                    else:
                        fallback_response = "I understand your question about customer complaints. While the AI system is experiencing technical difficulties, I can help you explore the analytics dashboard for insights, or you can try rephrasing your question."
                    
                    # Display both error and fallback
                    message_placeholder.error(error_msg)
                    st.warning("💡 **Fallback Response:** " + fallback_response)
                    
                    # Add combined message to chat
                    combined_message = f"{error_msg}\n\n💡 **Fallback Response:** {fallback_response}"
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": combined_message,
                        "sources": []
                    })

# Quick Actions
if st.session_state.messages:
    st.markdown("### 🚀 Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 New Conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    with col2:
        if st.button("📊 Show Analytics", use_container_width=True):
            st.session_state.show_analytics = True
            st.rerun()
    
    with col3:
        if st.button("💡 Sample Questions", use_container_width=True):
            st.session_state.show_sample_questions = True
            
            sample_questions = [
            "What are the most common issues with BNPL services?",
            "Which products have the highest complaint rates?",
            "What are customer satisfaction trends over time?",
            "How do complaint patterns vary by product category?"
            ]
        
            st.markdown("### 💡 Sample Questions to Try:")
            for i, question in enumerate(sample_questions, 1):
                st.markdown(f"{i}. **{question}**")

# Footer
st.markdown("---")

# Add some interactivity and insights
with st.sidebar: 
    if st.session_state.messages:
        st.markdown("### 📈 Chat Insights")
        
        # Generate insights based on chat history
        user_questions = [msg['content'] for msg in st.session_state.messages if msg['role'] == 'user']

        if user_questions:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.info(f"**💬 Total Questions:** {len(user_questions)}")
            
            with col2:
                st.success(f"**🔍 Current Filter:** {st.session_state.current_product_filter}")
            
            with col3:
                if len(user_questions) > 3:
                    st.warning("**🎯 Pro User** ")
                elif len(user_questions) > 1:
                    st.success("**📚 Keep exploring!**")
                else:
                    st.info("**🚀 Start:** Ask more!")

