from models.document_models import Document, DocumentChunk
from models.database import create_tables

create_tables()
print("Database tables created successfully.")
