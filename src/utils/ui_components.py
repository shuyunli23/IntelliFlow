"""
UI components for Streamlit interface
"""
import streamlit as st
from datetime import datetime
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class UIComponents:
    """UI components with natural text display"""
    
    @staticmethod
    def render_chat_stats(chat_history):
        """Render chat statistics in sidebar"""
        st.sidebar.header("💬 Chat History")
        stats = chat_history.get_stats()
        st.sidebar.info(f"Total messages: {stats['total_messages']}\nUser messages: {stats['user_messages']}")
        
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("📥 Export", use_container_width=True):
                csv = chat_history.export_to_csv()
                if csv:
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
        
        with col2:
            if st.button("🗑️ Clear", use_container_width=True):
                if chat_history.clear_history():
                    st.success("Chat history cleared!")
                    st.rerun()
    
    @staticmethod
    def render_enhanced_chat_history(chat_history):
        """Render enhanced chat history with natural formatting"""
        messages = chat_history.get_history()
        
        for message in messages:
            role = message.get('role', '')
            content = message.get('content', '')
            
            if role == "assistant_think":
                with st.expander("🧠 AI Thinking Process", expanded=False):
                    # Display as natural text, not markdown headers
                    st.text(content)  
            elif role == "retrieved_doc":
                with st.expander("📚 Retrieved Documents", expanded=False):
                    if isinstance(content, list):
                        for idx, doc in enumerate(content, 1):
                            with st.container():
                                st.markdown(doc)
                                if idx < len(content):
                                    st.divider()
                    else:
                        st.markdown(content)
            elif role == "weather_info":
                with st.expander("🌤️ Weather Information", expanded=False):
                    st.info(content)
            elif role == "execution_metadata":
                with st.expander("📊 Execution Summary", expanded=False):
                    st.markdown(content)
            else:
                with st.chat_message(role):
                    st.write(content)
    
    @staticmethod
    def render_chat_history(chat_history):
        """Backward compatibility method"""
        UIComponents.render_enhanced_chat_history(chat_history)