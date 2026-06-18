from repositories.document_repository import list_documents


def get_documents_status(limit: int) -> list[dict]:
    if limit <= 0:
        raise ValueError("limit must be greater than 0.")

    return list_documents(limit)
