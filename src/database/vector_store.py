"""
PostgreSQL + pgvector implementation for document storage and retrieval
"""
import logging
from typing import List, Optional
import json
import dashscope
from langchain.embeddings.base import Embeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy import select

from src.database.connection import get_db_session
from src.config.settings import settings
from src.utils.decorators import error_handler, log_execution
from src.database.models import Document as DocumentModel

logger = logging.getLogger(__name__)


class DashScopeEmbeddings(Embeddings):
    def __init__(self, api_key: str, model: str = "text-embedding-v2", max_batch_size: int = 25):
        dashscope.api_key = api_key
        self.model = model
        self.max_batch_size = max_batch_size
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs with batch size limit."""
        try:
            clean_texts = [str(text).strip() for text in texts if str(text).strip()]
            if not clean_texts:
                return []
            
            all_embeddings = []
            
            for i in range(0, len(clean_texts), self.max_batch_size):
                batch_texts = clean_texts[i:i + self.max_batch_size]
                
                try:
                    response = dashscope.TextEmbedding.call(
                        model=self.model,
                        input=batch_texts
                    )
                    
                    if response.status_code == 200:
                        batch_embeddings = [item['embedding'] for item in response.output['embeddings']]
                        all_embeddings.extend(batch_embeddings)
                        logger.info(f"Successfully embedded batch {i//self.max_batch_size + 1}, size: {len(batch_texts)}")
                    else:
                        logger.error(f"API error for batch {i//self.max_batch_size + 1}: {response}")
                        raise Exception(f"API error: {response}")
                        
                except Exception as e:
                    logger.error(f"Failed to embed batch {i//self.max_batch_size + 1}: {e}")
                    for text in batch_texts:
                        try:
                            single_response = dashscope.TextEmbedding.call(
                                model=self.model,
                                input=[text]
                            )
                            if single_response.status_code == 200:
                                single_embedding = single_response.output['embeddings'][0]['embedding']
                                all_embeddings.append(single_embedding)
                            else:
                                logger.error(f"Failed to embed single text: {single_response}")
                                continue
                        except Exception as single_e:
                            logger.error(f"Failed to embed single text: {single_e}")
                            continue
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Embedding documents failed: {e}")
            raise Exception(f"Embedding failed: {e}")
    
    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        try:
            clean_text = str(text).strip()
            if not clean_text:
                raise Exception("Empty query text")
                
            response = dashscope.TextEmbedding.call(
                model=self.model,
                input=[clean_text]
            )
            
            if response.status_code == 200:
                return response.output['embeddings'][0]['embedding']
            else:
                logger.error(f"Query embedding API error: {response}")
                raise Exception(f"API error: {response}")
                
        except Exception as e:
            logger.error(f"Embedding query failed: {e}")
            raise Exception(f"Embedding failed: {e}")


class PgVectorStore:
    """Enhanced PostgreSQL + pgvector implementation with fallback retrieval"""
    
    def __init__(self):
        """Initialize the vector store"""
        self.embeddings = DashScopeEmbeddings(
            api_key=settings.ali_api_key,
            model="text-embedding-v2",
            max_batch_size=20
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=settings.separators
        )
        
        logger.info("Enhanced PgVectorStore initialized successfully")
    
    @error_handler()
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        try:
            split_docs = self.text_splitter.split_documents(documents)
            logger.info(f"Document splitting completed: {len(documents)} -> {len(split_docs)} chunks")
            return split_docs
        except Exception as e:
            logger.error(f"Document splitting failed: {str(e)}")
            return documents
    
    @error_handler()
    @log_execution
    def add_documents(self, documents: List[Document]) -> bool:
        """Add documents to the vector store using ORM"""
        if not documents:
            logger.warning("No documents to add")
            return False
        
        try:
            split_docs = self.split_documents(documents)
            texts = [doc.page_content for doc in split_docs]
            embeddings = self.embeddings.embed_documents(texts)
            
            with get_db_session() as session:
                doc_objects = [
                    DocumentModel(
                        content=doc.page_content,
                        doc_metadata=doc.metadata,
                        embedding=embedding
                    ) for doc, embedding in zip(split_docs, embeddings)
                ]
                session.add_all(doc_objects)
            
            logger.info(f"Successfully added {len(split_docs)} document chunks")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add documents: {str(e)}")
            return False
    
    @error_handler()
    @log_execution
    def similarity_search_with_fallback(
        self, 
        query: str, 
        threshold: float = 0.7, 
        k: int = None,
        thinking_process: Optional[List[str]] = None
    ) -> List[Document]:
        """
        Enhanced similarity search with automatic fallback to top-k
        
        Args:
            query: Search query
            threshold: Similarity threshold for initial search
            k: Number of documents to return
            thinking_process: List to collect thinking process
            
        Returns:
            List of similar documents
        """
        if k is None:
            k = settings.max_retrieved_docs
        
        try:
            # Check if we have any documents
            total_docs = self.get_document_count()
            if total_docs == 0:
                if thinking_process is not None:
                    thinking_process.append("Database is empty")
                logger.info("No documents in database")
                return []
            
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # First attempt: threshold-based search
            with get_db_session() as session:
                stmt = select(
                    DocumentModel.content,
                    DocumentModel.doc_metadata,
                    (1 - DocumentModel.embedding.cosine_distance(query_embedding)).label("similarity")
                ).where(
                    (1 - DocumentModel.embedding.cosine_distance(query_embedding)) > threshold
                ).order_by(
                    DocumentModel.embedding.cosine_distance(query_embedding)
                ).limit(k)
                
                result = session.execute(stmt).fetchall()
            
            # If threshold search found results, use them
            if result:
                documents = self._convert_results_to_documents(result, thinking_process, "threshold-based")
                if thinking_process is not None:
                    thinking_process.append(f"Found {len(documents)} documents above threshold {threshold}")
                return documents
            
            # Fallback: top-k search without threshold
            if thinking_process is not None:
                thinking_process.append(f"No documents above threshold {threshold}, using top-{k} fallback")
            
            with get_db_session() as session:
                stmt = select(
                    DocumentModel.content,
                    DocumentModel.doc_metadata,
                    (1 - DocumentModel.embedding.cosine_distance(query_embedding)).label("similarity")
                ).order_by(
                    DocumentModel.embedding.cosine_distance(query_embedding)
                ).limit(k)
                
                result = session.execute(stmt).fetchall()
            
            documents = self._convert_results_to_documents(result, thinking_process, "top-k fallback")
            
            if thinking_process is not None:
                if documents:
                    thinking_process.append(f"Retrieved top {len(documents)} most similar documents")
                else:
                    thinking_process.append("No documents found")
            
            return documents
            
        except Exception as e:
            logger.error(f"Enhanced similarity search failed: {str(e)}")
            if thinking_process is not None:
                thinking_process.append(f"Search error: {str(e)}")
            return []
    
    def _convert_results_to_documents(
        self, 
        result, 
        thinking_process: Optional[List[str]], 
        search_type: str
    ) -> List[Document]:
        """Convert database results to Document objects with minimal analysis"""
        documents = []
        
        for i, row in enumerate(result, 1):
            metadata = row.doc_metadata if row.doc_metadata else {}
            similarity_score = float(row.similarity)
            metadata["similarity"] = similarity_score
            metadata["rank"] = i
            metadata["search_type"] = search_type
            
            documents.append(Document(
                page_content=row.content,
                metadata=metadata
            ))
        
        # Only add key summary to thinking process
        if thinking_process is not None and documents:
            avg_similarity = sum(doc.metadata["similarity"] for doc in documents) / len(documents)
            if avg_similarity > 0.8:
                quality = "Excellent"
            elif avg_similarity > 0.6:
                quality = "Good"
            elif avg_similarity > 0.4:
                quality = "Fair"
            else:
                quality = "Poor"
            
            thinking_process.append(f"Match quality: {quality} (avg similarity: {avg_similarity:.3f})")
        
        return documents
    
    @error_handler()
    @log_execution
    def similarity_search(
        self, 
        query: str, 
        threshold: float = 0.7, 
        k: int = None
    ) -> List[Document]:
        """
        Backward compatibility method - now uses enhanced search with fallback
        """
        return self.similarity_search_with_fallback(
            query=query, 
            threshold=threshold, 
            k=k
        )
    
    # Add the missing method that was referenced in nodes.py
    @error_handler()
    @log_execution
    def similarity_search_top_k(
        self, 
        query: str, 
        k: int = None,
        thinking_process: Optional[List[str]] = None
    ) -> List[Document]:
        """
        Direct top-k similarity search without threshold
        
        Args:
            query: Search query
            k: Number of top documents to return
            thinking_process: List to collect thinking process
            
        Returns:
            Top-k most similar documents
        """
        if k is None:
            k = settings.max_retrieved_docs
        
        try:
            # Check if we have any documents
            total_docs = self.get_document_count()
            if total_docs == 0:
                if thinking_process is not None:
                    thinking_process.append("Database is empty")
                return []
            
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Perform top-k search
            with get_db_session() as session:
                stmt = select(
                    DocumentModel.content,
                    DocumentModel.doc_metadata,
                    (1 - DocumentModel.embedding.cosine_distance(query_embedding)).label("similarity")
                ).order_by(
                    DocumentModel.embedding.cosine_distance(query_embedding)
                ).limit(k)
                
                result = session.execute(stmt).fetchall()
            
            documents = self._convert_results_to_documents(result, thinking_process, "top-k")
            logger.info(f"Retrieved top {len(documents)} documents")
            return documents
            
        except Exception as e:
            logger.error(f"Top-k similarity search failed: {str(e)}")
            if thinking_process is not None:
                thinking_process.append(f"Search error: {str(e)}")
            return []
    
    @error_handler()
    def get_context(self, documents: List[Document]) -> str:
        """Get context from documents with enhanced formatting"""
        if not documents:
            return ""
        
        context_parts = []
        for doc in documents:
            rank = doc.metadata.get("rank", "?")
            similarity = doc.metadata.get("similarity", 0)
            source = doc.metadata.get("source", "Unknown")
            search_type = doc.metadata.get("search_type", "unknown")
            
            header = f"[Document {rank} - {source} - Similarity: {similarity:.3f} - Method: {search_type}]"
            context_parts.append(f"{header}\n{doc.page_content}")
        
        return "\n\n".join(context_parts)
    
    @error_handler()
    def clear_documents(self) -> bool:
        """Clear all documents from the vector store using ORM"""
        try:
            with get_db_session() as session:
                session.query(DocumentModel).delete()
            logger.info("All documents cleared successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to clear documents: {str(e)}")
            return False
    
    @error_handler()
    def get_document_count(self) -> int:
        """Get total number of documents using ORM"""
        try:
            with get_db_session() as session:
                return session.query(DocumentModel).count()
        except Exception as e:
            logger.error(f"Failed to get document count: {str(e)}")
            return 0