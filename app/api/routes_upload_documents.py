from pathlib import Path

from fastapi import APIRouter, File, HTTPException,UploadFile,status
from services.process_uploaded_document import process_uploaded_document
from services.document_parser import SUPPORTED_DOCUMENT_EXTENSIONS

router=APIRouter(prefix='/documents',tags=['documents'])

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPLOAD_DIR=PROJECT_ROOT / "data" / "uploads"

@router.post("/upload",status_code=status.HTTP_201_CREATED)
async def upload_document(file:UploadFile=File(...)):
    filename=Path(file.filename).name
    file_ext=Path(filename).suffix.lower()
    if file_ext not in SUPPORTED_DOCUMENT_EXTENSIONS:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,detail=f"Unsupported file type {file_ext}")
    file_content=await file.read()
    if not file_content:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,detail="File is empty")
    UPLOAD_DIR.mkdir(parents=True,exist_ok=True)
    file_path=UPLOAD_DIR/filename
    file_path.write_bytes(file_content)
    file_info={'filename': filename,
     'content_type': file.content_type,
     'saved_path': str(file_path),
     'file_size': len(file_content),}
    try:
        processed_document=process_uploaded_document(file_path,file_info)
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,detail=str(e))

    except UnicodeDecodeError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,detail="Could not decode document as UTF-8 text.",)

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,detail=str(e))

    return {**file_info,**processed_document}
