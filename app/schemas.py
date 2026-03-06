from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    chat_id: str
    question: str
    top_k: int = 5


class Chunk(BaseModel):
    text: str
    source: str | None = None
    score: float | None = None


class ChatResponse(BaseModel):
    answer: str
    model: str = "gpt-4o-mini"
    chunks: list[Chunk] = Field(default_factory=list)