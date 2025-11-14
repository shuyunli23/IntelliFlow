# src/database/models.py
from sqlalchemy import Column, Integer, Text, JSON, TIMESTAMP, func
from sqlalchemy.orm import declarative_base, mapped_column
from pgvector.sqlalchemy import Vector

# 创建 Base
Base = declarative_base()

class Document(Base):
    __tablename__ = 'documents'
    
    id = Column(Integer, primary_key=True)
    content = Column(Text, nullable=False)
    # 将 metadata 改为 doc_metadata 避免与 SQLAlchemy 的保留字段冲突
    doc_metadata = Column(JSON, name='metadata')  # 数据库中仍然叫 metadata
    
    # 关键：使用 Vector 类型定义 embedding 字段
    # DashScope text-embedding-v2 模型的维度是 1536
    embedding = mapped_column(Vector(1536))
    
    created_at = Column(TIMESTAMP, server_default=func.now())


class ChatHistory(Base):
    __tablename__ = 'chat_history'

    id = Column(Integer, primary_key=True)
    role = Column(Text, nullable=False)
    content = Column(Text, nullable=False)
    # 同样修改这里的 metadata
    chat_metadata = Column(JSON, name='metadata')  # 数据库中仍然叫 metadata
    created_at = Column(TIMESTAMP, server_default=func.now())