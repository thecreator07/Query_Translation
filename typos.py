from typing import Optional
from pydantic import BaseModel

class Question(BaseModel):
    question: str
    answer: Optional[str] = None

class Questions(BaseModel):
    questions: list[Question]