from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.api.error_handlers import map_exception_to_http
from services.rag_query_service import answer_question, semantic_search


class SearchQuestion(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


router = APIRouter(prefix="/query", tags=["query"])


@router.post("/search")
def search_api(search_question: SearchQuestion):
    try:
        question = search_question.question.strip()
        if not question:
            raise ValueError("Question cannot be empty.")

        return semantic_search(question)

    except Exception as e:
        print(e)
        raise map_exception_to_http(e)


@router.post("/answer")
def answer_api(search_question: SearchQuestion):
    try:
        question = search_question.question.strip()
        if not question:
            raise ValueError("Question cannot be empty.")

        return answer_question(question)

    except Exception as e:
        print(e)
        raise map_exception_to_http(e)
