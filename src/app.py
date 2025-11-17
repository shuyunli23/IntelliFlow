"""
IntelliFlow Streamlit Application
"""
import streamlit as st
import logging
from datetime import datetime
from typing import List, Dict, Any
import re

from src.config.settings import settings
from src.database.connection import init_database, test_connection
from src.database.vector_store import PgVectorStore
from src.agents.graph import IntelliFlowGraph
from src.tools.document_processor import DocumentProcessor
from src.utils.chat_history import ChatHistoryManager
from src.utils.ui_components import UIComponents
from src.utils.decorators import error_handler, log_execution

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntelliFlowApp:
    """Enhanced IntelliFlow application with formatted thinking process"""
    
    def __init__(self):
        """Initialize the application"""
        self._init_session_state()
        self.chat_history = ChatHistoryManager()
        self.document_processor = DocumentProcessor()
        self.vector_store = PgVectorStore()
        self.agent_graph = IntelliFlowGraph()
        
        # Initialize database
        if not test_connection():
            st.error("❌ Database connection failed. Please check your PostgreSQL configuration.")
            st.stop()
        
        init_database()
        logger.info("Enhanced IntelliFlow application initialized successfully")
    
    @error_handler(show_error=False)
    def _init_session_state(self):
        """Initialize Streamlit session state"""
        if 'model_version' not in st.session_state:
            st.session_state.model_version = settings.ali_model_name
        if 'processed_documents' not in st.session_state:
            st.session_state.processed_documents = []
        if 'similarity_threshold' not in st.session_state:
            st.session_state.similarity_threshold = settings.default_similarity_threshold
        if 'rag_enabled' not in st.session_state:
            st.session_state.rag_enabled = True
        if 'show_thinking' not in st.session_state:
            st.session_state.show_thinking = True
    
    @error_handler()
    @log_execution
    def render_sidebar(self):
        """Enhanced sidebar with thinking process toggle"""
        st.sidebar.header("⚙️ Settings")
        
        # Model selection
        st.session_state.model_version = st.sidebar.selectbox(
            "Select Model",
            options=settings.available_models,
            index=settings.available_models.index(st.session_state.model_version) 
            if st.session_state.model_version in settings.available_models else 0,
            help="Choose the language model to use"
        )
        
        # RAG settings
        st.sidebar.subheader("📚 RAG Settings")
        
        st.session_state.rag_enabled = st.sidebar.checkbox(
            "Enable RAG",
            value=st.session_state.rag_enabled,
            help="Enable Retrieval-Augmented Generation using uploaded documents"
        )
        
        st.session_state.similarity_threshold = st.sidebar.slider(
            "Similarity Threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.similarity_threshold,
            step=0.05,
            help="Similarity threshold with automatic fallback to top-k"
        )
        
        if st.sidebar.button("Reset Threshold", use_container_width=True):
            st.session_state.similarity_threshold = settings.default_similarity_threshold
            st.rerun()
        
        # Info about fallback mechanism
        st.sidebar.info("🎯 **Smart Retrieval**\nIf no documents meet the threshold, automatically retrieves the most similar ones.")
        
        # Display settings
        st.sidebar.subheader("🖥️ Display Settings")
        
        st.session_state.show_thinking = st.sidebar.checkbox(
            "Show Thinking Process",
            value=st.session_state.show_thinking,
            help="Display the AI's detailed reasoning process"
        )
        
        # Database info
        st.sidebar.subheader("📊 Database Info")
        doc_count = self.vector_store.get_document_count()
        st.sidebar.info(f"Documents in database: {doc_count}")
        
        # Chat statistics
        UIComponents.render_chat_stats(self.chat_history)
    
    @error_handler()
    @log_execution
    def render_document_upload(self):
        """Render document upload interface"""
        with st.expander("📁 Upload Documents for RAG", expanded=not bool(st.session_state.processed_documents)):
            uploaded_files = st.file_uploader(
                "Upload PDF or TXT files",
                type=["pdf", "txt"],
                accept_multiple_files=True,
                help="Upload documents to enhance the AI's knowledge base"
            )
            
            if uploaded_files:
                if st.button("Process Documents", type="primary"):
                    with st.spinner("Processing documents..."):
                        success_count = 0
                        for uploaded_file in uploaded_files:
                            if uploaded_file.name not in st.session_state.processed_documents:
                                try:
                                    documents = self.document_processor.process_file(uploaded_file)
                                    
                                    if self.vector_store.add_documents(documents):
                                        st.session_state.processed_documents.append(uploaded_file.name)
                                        success_count += 1
                                        st.success(f"✅ Processed: {uploaded_file.name}")
                                    else:
                                        st.error(f"❌ Failed to add to database: {uploaded_file.name}")
                                        
                                except Exception as e:
                                    st.error(f"❌ Processing failed: {uploaded_file.name} - {str(e)}")
                            else:
                                st.warning(f"⚠️ Already processed: {uploaded_file.name}")
                        
                        if success_count > 0:
                            st.success(f"🎉 Successfully processed {success_count} documents!")
            
            # Show processed documents
            if st.session_state.processed_documents:
                st.subheader("📚 Processed Documents")
                for doc in st.session_state.processed_documents:
                    st.markdown(f"- {doc}")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🗑️ Clear All Documents", use_container_width=True):
                        if self.vector_store.clear_documents():
                            st.session_state.processed_documents.clear()
                            st.success("✅ All documents cleared!")
                            st.rerun()
                        else:
                            st.error("❌ Failed to clear documents")
    
    @error_handler()
    @log_execution
    def process_user_input(self, prompt: str):
        """Enhanced user input processing"""
        # Add user message to history
        self.chat_history.add_message("user", prompt)
        
        with st.spinner("🤔 Processing your request..."):
            # Run the agent graph
            result = self.agent_graph.run(
                query=prompt,
                use_rag=st.session_state.rag_enabled,
                similarity_threshold=st.session_state.similarity_threshold
            )
            
            # Process the response
            self._process_response(result)
    
    def _process_response(self, result: Dict[str, Any]):
        """Enhanced response processing with formatted thinking"""
        response = result.get("response", "")
        formatted_thinking = result.get("formatted_thinking", "")
        retrieved_docs = result.get("retrieved_docs", [])
        weather_info = result.get("weather_info", "")
        execution_metadata = result.get("execution_metadata", {})
        
        # Save response to history
        self.chat_history.add_message("assistant", response)
        
        # Save formatted thinking process if enabled
        if formatted_thinking and st.session_state.show_thinking:
            self.chat_history.add_message("assistant_think", formatted_thinking)
        
        # Save retrieved documents with enhanced information
        if retrieved_docs:
            doc_contents = []
            for doc in retrieved_docs:
                source = doc.metadata.get("source", "Unknown")
                similarity = doc.metadata.get("similarity", 0)
                rank = doc.metadata.get("rank", "?")
                search_type = doc.metadata.get("search_type", "unknown")
                content = doc.page_content
                
                doc_info = f"**Document {rank}** - {source} ({search_type})\n"
                doc_info += f"**Similarity Score**: {similarity:.3f}\n"
                doc_info += f"**Content**: {content}"
                
                doc_contents.append(doc_info)
            
            self.chat_history.add_message("retrieved_doc", doc_contents)
        
        # Save weather info if available
        if weather_info:
            self.chat_history.add_message("weather_info", weather_info)
        
        # Save execution metadata
        if execution_metadata:
            metadata_text = f"""**Execution Summary**:
- Total attempts: {execution_metadata.get('total_loops', 1)}
- Query rewritten: {'Yes' if execution_metadata.get('query_rewritten') else 'No'}
- Reflection performed: {'Yes' if execution_metadata.get('reflection_performed') else 'No'}
- Final query type: {execution_metadata.get('final_query_type', 'Unknown')}"""
            
            self.chat_history.add_message("execution_metadata", metadata_text)
    
    @error_handler()
    @log_execution
    def run(self):
        """Run the enhanced application"""
        # Page configuration
        st.set_page_config(
            page_title="IntelliFlow Enhanced",
            page_icon="📡",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Main title
        st.title("📡 IntelliFlow - Enhanced RAG with Smart Retrieval")
        st.markdown("""
        **IntelliFlow Enhanced** features intelligent threshold-based retrieval with automatic 
        fallback to top-k search, ensuring you always get the most relevant information available.
        """)
        
        # Render sidebar
        self.render_sidebar()
        
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Document upload
            self.render_document_upload()
            
            # Chat interface
            st.subheader("💬 Enhanced Chat Interface")
            
            # Chat input
            prompt = st.chat_input(
                "Ask about your documents..." if st.session_state.rag_enabled 
                else "Ask me anything..."
            )
            
            if prompt:
                self.process_user_input(prompt)
            
            # Render enhanced chat history
            UIComponents.render_enhanced_chat_history(self.chat_history)
        
        with col2:
            # System status
            st.subheader("🔧 Enhanced System Status")
            
            # Database connection status
            if test_connection():
                st.success("✅ Database Connected")
            else:
                st.error("❌ Database Disconnected")
            
            # Current mode
            mode = "Smart RAG Mode" if st.session_state.rag_enabled else "Chat Mode"
            st.info(f"🤖 Current Mode: {mode}")
            
            # Model info
            st.info(f"🧠 Model: {st.session_state.model_version}")
            
            # Retrieval method
            st.success(f"🎯 Retrieval: Threshold {st.session_state.similarity_threshold} + Fallback")
            
            # Features
            st.subheader("✨ Enhanced Features")
            st.markdown("""
            - 📚 **Smart RAG**: Threshold-based with automatic fallback
            - 🧠 **Formatted Thinking**: Clean, readable reasoning process
            - 🔍 **Quality Reflection**: Self-evaluation and improvement
            - 🌤️ **Weather Integration**: Real-time weather information
            - 🔄 **Adaptive Retry**: Automatic quality improvement
            - 📊 **Detailed Analytics**: Complete execution tracking
            - 🎯 **Never Empty**: Always finds the most relevant information
            """)


if __name__ == "__main__":
    app = IntelliFlowApp()
    app.run()
