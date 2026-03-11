import streamlit as st
import asyncio
import logging
from typing import Optional
import os
from dotenv import load_dotenv, find_dotenv

from src.engine import SupportEngine

# Load environment variables
load_dotenv(find_dotenv())

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Support Ticket RAG System",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .ticket-card {
        background-color: #f8f9fa;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    .metadata-badge {
        display: inline-block;
        background-color: #e9ecef;
        padding: 0.25rem 0.5rem;
        margin: 0.25rem;
        border-radius: 3px;
        font-size: 0.85rem;
    }
    .tag-badge {
        display: inline-block;
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 0.25rem 0.5rem;
        margin: 0.25rem;
        border-radius: 3px;
        font-size: 0.85rem;
    }
    </style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if 'engine' not in st.session_state:
        st.session_state.engine = None
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []

def initialize_engine(force_reload: bool = False) -> Optional[SupportEngine]:
    """Initialize the support engine."""
    try:
        with st.spinner("Initializing Support Engine..."):
            if st.session_state.engine is None:
                st.session_state.engine = SupportEngine(
                    data_path="data",
                    persist_directory="vector_store"
                )
            
            if not st.session_state.initialized or force_reload:
                st.session_state.engine.initialize(force_reload=force_reload)
                st.session_state.initialized = True
                st.success("✅ Support Engine initialized successfully!")
            
            return st.session_state.engine
    except Exception as e:
        st.error(f"❌ Error initializing engine: {str(e)}")
        logger.error(f"Engine initialization error: {e}", exc_info=True)
        return None

def display_ticket_card(ticket: dict, index: int):
    """Display a ticket in a formatted card."""
    metadata = ticket.get('metadata', {})
    content = ticket.get('content', '')
    
    with st.container():
        st.markdown(f"""
        <div class="ticket-card">
            <h4>🎫 Ticket {index + 1}: {metadata.get('ticket_id', 'Unknown')}</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Metadata section
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**Support Type:** {metadata.get('support_type', 'N/A').title()}")
            st.markdown(f"**Priority:** {metadata.get('priority', 'N/A')}")
        with col2:
            st.markdown(f"**Type:** {metadata.get('type', 'N/A')}")
            st.markdown(f"**Queue:** {metadata.get('queue', 'N/A')}")
        with col3:
            st.markdown(f"**Language:** {metadata.get('language', 'N/A')}")
            st.markdown(f"**Source:** {metadata.get('source', 'N/A')}")
        
        # Tags
        tags = metadata.get('tags', [])
        if tags:
            st.markdown("**Tags:**")
            tags_html = " ".join([f'<span class="tag-badge">{tag}</span>' for tag in tags])
            st.markdown(tags_html, unsafe_allow_html=True)
        
        # Content
        with st.expander("📄 View Ticket Content", expanded=False):
            st.text(content)
        
        st.markdown("---")

async def run_query(query: str, support_type: str, k: int):
    """Run a query against the RAG system."""
    try:
        engine = st.session_state.engine
        rag_chain = engine.get_rag_chain()
        
        # Get relevant documents
        with st.spinner("🔍 Searching for relevant tickets..."):
            docs = rag_chain.get_relevant_documents(
                query=query,
                support_type=support_type,
                k=k,
            )
            print("docs==========>>>", docs)

        # Display relevant tickets
        if docs:
            st.subheader(f"📋 Found {len(docs)} Relevant Ticket(s)")
            for idx, doc in enumerate(docs):
                display_ticket_card(doc, idx)
        else:
            st.warning("⚠️ No relevant tickets found.")
            return
        
        # Generate AI response
        with st.spinner("🤖 Generating AI response..."):
            response = await rag_chain.query(
                query=query,
                support_type=support_type
            )
            print("response==========>>>", response)
        
        # Display AI response
        st.subheader("💡 AI Assistant Response")
        st.markdown(f"""
        <div style="background-color: #e8f4f8; color: black; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #1f77b4;">
            {response}
        </div>
        """, unsafe_allow_html=True)
        
        # Add to history
        st.session_state.query_history.append({
            'query': query,
            'support_type': support_type,
            'num_results': len(docs),
            'response': response
        })
        
    except ValueError as ve:
        st.error(f"❌ Validation Error: {str(ve)}")
    except Exception as e:
        st.error(f"❌ Error processing query: {str(e)}")
        logger.error(f"Query processing error: {e}", exc_info=True)

def main():
    """Main application function."""
    initialize_session_state()
    
    # Header
    st.markdown('<div class="main-header">🎫 Support Ticket RAG System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Intelligent Support Ticket Retrieval & Analysis</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key check
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            st.success("✅ OpenAI API Key loaded")
        else:
            st.error("❌ OpenAI API Key not found")
            st.info("Please set OPENAI_API_KEY in your .env file")
            return
        
        st.markdown("---")
        
        # Engine initialization
        st.subheader("🚀 Engine Status")
        if st.session_state.initialized:
            st.success("✅ Engine Ready")
            if st.button("🔄 Reload Data", help="Force reload all documents"):
                initialize_engine(force_reload=True)
        else:
            if st.button("▶️ Initialize Engine", type="primary"):
                initialize_engine()
        
        st.markdown("---")
        
        # Query settings
        st.subheader("🔧 Query Settings")
        support_type = st.selectbox(
            "Support Type",
            options=["Technical", "Product", "Customer"],
            help="Filter by support type"
        )
        
        num_results = st.slider(
            "Number of Results",
            min_value=1,
            max_value=10,
            value=3,
            help="Number of similar tickets to retrieve"
        )
        
        st.markdown("---")
        
        # Statistics
        if st.session_state.query_history:
            st.subheader("📊 Statistics")
            st.metric("Total Queries", len(st.session_state.query_history))
            if st.button("🗑️ Clear History"):
                st.session_state.query_history = []
                st.rerun()
    
    # Main content area
    if not st.session_state.initialized:
        st.info("👈 Please initialize the engine from the sidebar to get started.")
        
        # Display system information
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### 🎯 Features")
            st.markdown("""
            - Semantic search across tickets
            - Multi-format support (JSON, XML)
            - AI-powered responses
            - Support type filtering
            """)
        with col2:
            st.markdown("### 📚 Data Sources")
            st.markdown("""
            - Technical Support
            - Product Support
            - Customer Service
            """)
        with col3:
            st.markdown("### 🔧 Technologies")
            st.markdown("""
            - LangChain
            - ChromaDB
            - OpenAI GPT-4
            - Streamlit
            """)
        return
    
    # Query interface
    st.subheader("🔍 Search Support Tickets")
    
    query = st.text_area(
        "Enter your query:",
        placeholder="e.g., How do I reset my password?",
        height=100,
        help="Describe your support issue or question"
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    with col2:
        if st.session_state.query_history:
            st.caption(f"Last query: {st.session_state.query_history[-1]['query'][:50]}...")
    
    if search_button:
        if not query or len(query.strip()) < 10:
            st.warning("⚠️ Please enter a query with at least 10 characters.")
        else:
            # Convert support type to lowercase for API
            support_type_param = support_type.lower()
            
            # Run async query
            asyncio.run(run_query(query, support_type_param, num_results))
    
    # Query history
    if st.session_state.query_history:
        st.markdown("---")
        with st.expander("📜 Query History", expanded=False):
            for idx, item in enumerate(reversed(st.session_state.query_history[-5:])):
                st.markdown(f"**Query {len(st.session_state.query_history) - idx}:** {item['query']}")
                st.caption(f"Support Type: {item['support_type']} | Results: {item['num_results']}")
                st.markdown("---")

if __name__ == "__main__":
    main()
