from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import time

from app.rag import rag_context
from app.schemas import ChatRequest, ChatResponse
from app.llm import run_llm
from app.src.memory import get_history, add_message, history_to_text

load_dotenv()

app = FastAPI(title="LLM API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    print("health check")
    return {"status": "ok", "ts": int(time.time())}

@app.post("/v1/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    chat_id = req.chat_id
    question = req.question

    history = get_history(chat_id)
    history_text = history_to_text(history)
    context = rag_context(question)

    messages = [{"role": "system", "content": f"Контекст:\n{context}"}]
    messages += history
    messages += [{"role": "user", "content": question}]

    answer = run_llm(context, question, history_text)

    add_message(chat_id, "user", question)
    add_message(chat_id, "assistant", answer)

    return ChatResponse(answer=answer)