"""
Document processing utilities
"""
import logging
from typing import List, Union
import tempfile
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from src.utils.decorators import error_handler, log_execution

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Document processor for various file types"""
    
    def __init__(self):
        """Initialize document processor"""
        logger.info("Document processor initialized")
    
    @error_handler()
    @log_execution
    def process_file(self, uploaded_file) -> List[Document]:
        """
        Process uploaded file and return documents
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            List of Document objects
        """
        file_content = uploaded_file.getvalue()
        file_name = uploaded_file.name
        
        if file_name.lower().endswith('.pdf'):
            return self._process_pdf(file_content, file_name)
        elif file_name.lower().endswith('.txt'):
            return self._process_txt(file_content, file_name)
        else:
            raise ValueError(f"Unsupported file type: {file_name}")
    
    def _process_pdf(self, file_content: bytes, file_name: str) -> List[Document]:
        """Process PDF file"""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_file.write(file_content)
            temp_file.flush()
            
            try:
                loader = PyPDFLoader(temp_file.name)
                documents = loader.load()
                
                # Add source metadata
                for doc in documents:
                    doc.metadata["source"] = file_name
                
                return documents
                
            finally:
                Path(temp_file.name).unlink(missing_ok=True)
    
    def _process_txt(self, file_content: bytes, file_name: str) -> List[Document]:
        """Process TXT file"""
        content = file_content.decode('utf-8')
        return [Document(
            page_content=content,
            metadata={"source": file_name}
        )]
    