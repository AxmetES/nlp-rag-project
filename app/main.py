from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import time

from rag import rag_context, get_vectorstore, is_index_ready
from schemas import ChatRequest, ChatResponse
from llm import run_llm
from src.build_index import build_index
from src.memory import get_history, add_message, history_to_text
from src.utils import docs_to_context, docs_to_chunks

load_dotenv()

app = FastAPI(title="LLM API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    try:
        vs = get_vectorstore()

        # 1) Если индекс существует и не пустой -> просто работаем
        if is_index_ready(vs):
            print("✅ Vector DB ready.")
            app.state.vectorstore = vs
            return

        # 2) Индекс пустой (обычно первый запуск) -> строим
        print("⚠️ Vector DB empty. Building index...")
        build_index(force=False)

        vs = get_vectorstore()
        if not is_index_ready(vs):
            raise RuntimeError("Index build finished, but DB is still empty.")

        print("✅ Index built.")
        app.state.vectorstore = vs

    except Exception as e:
        # 3) Если база битая/несовместимая -> пересоздаём
        print(f"❌ Vector DB error: {e}")
        print("⚠️ Rebuilding index from scratch...")

        build_index(force=True)  # вот тут force=True оправдан

        vs = get_vectorstore()
        app.state.vectorstore = vs
        print("✅ Index rebuilt.")


@app.get("/health")
def health():
    print("health check")
    return {"status": "ok", "ts": int(time.time())}

@app.post("/v1/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    chat_id = req.chat_id
    question = req.question
    top_k = req.top_k

    history = get_history(chat_id)
    history_text = history_to_text(history)
    vs = app.state.vectorstore
    docs_score = rag_context(question, vs, k=top_k)

    context = docs_to_context([doc for doc, _ in docs_score])
    messages = [{"role": "system", "content": f"Контекст:\n{context}"}]
    messages += history
    messages += [{"role": "user", "content": question}]

    answer = run_llm(context, question, history_text)

    add_message(chat_id, "user", question)
    add_message(chat_id, "assistant", answer)
    chunks = docs_to_chunks(docs_score)
    return ChatResponse(answer=answer, chunks=chunks)