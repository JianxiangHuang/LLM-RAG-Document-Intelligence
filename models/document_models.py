from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB
from models.database import Base


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True)
    filename = Column(String(512), nullable=False, index=True)
    saved_path = Column(Text, nullable=False)
    content_type = Column(String(512), nullable=True)
    file_size = Column(Integer, nullable=False)
    status = Column(String(50), nullable=False, default="uploaded")
    char_count = Column(Integer, nullable=False)
    chunk_count = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    file_hash = Column(String(64), nullable=False, index=True)
    content_hash = Column(String(64), nullable=True, index=True)
    chunks = relationship("DocumentChunk", back_populates="document")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "filename": self.filename,
            "saved_path": self.saved_path,
            "content_type": self.content_type,
            "file_size": self.file_size,
            "status": self.status,
            "char_count": self.char_count,
            "chunk_count": self.chunk_count,
            "created_at": self.created_at,
            "file_hash": self.file_hash,
            "content_hash": self.content_hash,
        }



class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True)
    document_id = Column(Integer,ForeignKey("documents.id"),nullable=False,index=True,)
    chunk_index = Column(Integer, nullable=False)  
    chunk_method=Column(String(50), nullable=False,default="fixed",)
    chunk_type=Column(String(50), nullable=False,default="small",)
    parent_chunk_id = Column(Integer,ForeignKey("document_chunks.id",ondelete="SET NULL"), nullable=True,index=True)
    text = Column(Text, nullable=False)
    contextual_header=Column(Text, nullable=True)
    char_count = Column(Integer, nullable=False)
    embedding = Column(Vector(1536), nullable=True)
    source_position = Column(JSONB,nullable=True,)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    document = relationship("Document", back_populates="chunks")
    parent_chunk = relationship("DocumentChunk",remote_side=[id],backref="child_chunks",)
