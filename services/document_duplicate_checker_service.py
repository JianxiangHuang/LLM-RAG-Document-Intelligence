import hashlib

from repositories.document_repository import find_document_by_file_hash,find_document_by_filename

def calculate_file_hash(file_content: bytes)->str:

    if not file_content:
        raise ValueError("File content cannot be empty.")

    file_hash = hashlib.sha256(file_content).hexdigest()
    return file_hash

def is_file_duplicate(file_hash:str)->bool:
    return find_document_by_file_hash(file_hash) is not None

def is_filename_duplicate(filename:str)->bool:
    return find_document_by_filename(filename) is not None