from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile, status

from app.api.error_handlers import map_exception_to_http
from services.process_uploaded_document import process_uploaded_document, save_uploaded_document_to_db
from services.document_parser import SUPPORTED_DOCUMENT_EXTENSIONS
from services.document_status_query_service import get_documents_status

router = APIRouter(prefix="/documents", tags=["documents"])

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"


@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    filename = Path(file.filename or "").name
    if not filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required.",
        )

    file_ext = Path(filename).suffix.lower()
    if file_ext not in SUPPORTED_DOCUMENT_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type {file_ext}",
        )

    file_content = await file.read()
    if not file_content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File is empty",
        )

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    file_path = UPLOAD_DIR / filename
    file_path.write_bytes(file_content)

    file_info = {
        "filename": filename,
        "content_type": file.content_type,
        "saved_path": str(file_path),
        "file_size": len(file_content),
    }

    try:
        document_id = save_uploaded_document_to_db(file_info)
        background_tasks.add_task(process_uploaded_document, document_id)
    except Exception as e:
        raise map_exception_to_http(e)

    return {
        **file_info,
        "document_id": document_id,
        "status": "uploaded",
    }


@router.post("/status")
async def get_all_documents_status(limit: int = 20):
    try:
        documents = get_documents_status(limit)
        return {
            "count": len(documents),
            "limit": limit,
            "documents": [
                {
                    "id": document["id"],
                    "filename": document["filename"],
                    "content_type": document["content_type"],
                    "file_size": document["file_size"],
                    "status": document["status"],
                    "chunk_count": document["chunk_count"],
                    "created_at": (
                        document["created_at"].isoformat()
                        if document["created_at"]
                        else None
                    ),
                }
                for document in documents
            ]
        }
    except Exception as e:
        raise map_exception_to_http(e)
