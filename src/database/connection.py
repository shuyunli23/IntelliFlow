"""
Database connection management
"""
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from contextlib import contextmanager
from typing import Generator
from src.config.settings import settings

# 导入我们新创建的模型中的 Base
from src.database.models import Base 

logger = logging.getLogger(__name__)

# Create database engine
engine = create_engine(
    settings.database_url,
    poolclass=StaticPool,
    pool_pre_ping=True,
    echo=False
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@contextmanager
def get_db_session() -> Generator:
    """
    Get database session with automatic cleanup
    
    Yields:
        Session: Database session
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database session error: {str(e)}")
        raise
    finally:
        session.close()


def init_database():
    """Initialize database tables using SQLAlchemy's ORM capabilities"""
    try:
        with engine.connect() as connection:
            # 首先确保 pgvector 扩展存在
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            connection.commit()
        
        # 使用 Base.metadata.create_all() 来创建所有继承自 Base 的表
        Base.metadata.create_all(bind=engine)
        
        # 创建索引
        with get_db_session() as session:
            session.execute(text("""
                CREATE INDEX IF NOT EXISTS documents_embedding_idx 
                ON documents USING ivfflat (embedding vector_cosine_ops) 
                WITH (lists = 100);
            """))
            
        logger.info("Database initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")
        raise


def test_connection() -> bool:
    """
    Test database connection
    
    Returns:
        bool: True if connection is successful
    """
    try:
        with get_db_session() as session:
            session.execute(text("SELECT 1"))
        logger.info("Database connection test successful")
        return True
    except Exception as e:
        logger.error(f"Database connection test failed: {str(e)}")
        return False
    