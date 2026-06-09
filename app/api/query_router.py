from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

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

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/answer")
def answer_api(search_question: SearchQuestion):
    try:
        question = search_question.question.strip()
        if not question:
            raise ValueError("Question cannot be empty.")

        return answer_question(question)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")
