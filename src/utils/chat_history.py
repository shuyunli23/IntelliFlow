"""
Chat history management using PostgreSQL
"""
import logging
from typing import List, Dict, Optional
import json
from datetime import datetime
from sqlalchemy import text
from src.database.connection import get_db_session
from src.config.settings import settings

logger = logging.getLogger(__name__)


class ChatHistoryManager:
    """Chat history manager using PostgreSQL"""
    
    def __init__(self):
        """Initialize chat history manager"""
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info("Chat history manager initialized")
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None) -> bool:
        """
        Add message to chat history
        
        Args:
            role: Message role (user/assistant/system)
            content: Message content
            metadata: Optional metadata
            
        Returns:
            bool: True if successful
        """
        try:
            with get_db_session() as session:
                session.execute(
                    text("""
                        INSERT INTO chat_history (role, content, metadata)
                        VALUES (:role, :content, :metadata)
                    """),
                    {
                        "role": role,
                        "content": content,
                        "metadata": json.dumps(metadata or {})
                    }
                )
            return True
            
        except Exception as e:
            logger.error(f"Failed to add message: {str(e)}")
            return False
    
    def get_history(self, limit: int = None) -> List[Dict]:
        """
        Get chat history
        
        Args:
            limit: Maximum number of messages to retrieve
            
        Returns:
            List of chat messages
        """
        if limit is None:
            limit = settings.max_history_turns * 2
        
        try:
            with get_db_session() as session:
                result = session.execute(
                    text("""
                        SELECT role, content, metadata, created_at
                        FROM chat_history
                        ORDER BY created_at DESC
                        LIMIT :limit
                    """),
                    {"limit": limit}
                ).fetchall()
            
            # Reverse to get chronological order
            messages = []
            for row in reversed(result):
                metadata = json.loads(row.metadata) if row.metadata else {}
                messages.append({
                    "role": row.role,
                    "content": row.content,
                    "metadata": metadata,
                    "created_at": row.created_at
                })
            
            return messages
            
        except Exception as e:
            logger.error(f"Failed to get history: {str(e)}")
            return []
    
    def clear_history(self) -> bool:
        """
        Clear all chat history
        
        Returns:
            bool: True if successful
        """
        try:
            with get_db_session() as session:
                session.execute(text("DELETE FROM chat_history"))
            
            logger.info("Chat history cleared successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear history: {str(e)}")
            return False
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get chat history statistics
        
        Returns:
            Dictionary with statistics
        """
        try:
            with get_db_session() as session:
                result = session.execute(
                    text("""
                        SELECT 
                            COUNT(*) as total_messages,
                            COUNT(CASE WHEN role = 'user' THEN 1 END) as user_messages
                        FROM chat_history
                    """)
                ).fetchone()
            
            return {
                "total_messages": result.total_messages if result else 0,
                "user_messages": result.user_messages if result else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to get stats: {str(e)}")
            return {"total_messages": 0, "user_messages": 0}
    
    def export_to_csv(self) -> Optional[bytes]:
        """
        Export chat history to CSV
        
        Returns:
            CSV data as bytes or None if failed
        """
        try:
            import pandas as pd
            
            messages = self.get_history(limit=1000)  # Get more messages for export
            if not messages:
                return None
            
            df = pd.DataFrame(messages)
            return df.to_csv(index=False).encode('utf-8')
            
        except Exception as e:
            logger.error(f"Failed to export to CSV: {str(e)}")
            return None
        