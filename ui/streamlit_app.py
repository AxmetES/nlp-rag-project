import os
import requests
import streamlit as st

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

API_URL = os.getenv("API_URL", "http://localhost:8000/v1/chat")

st.title("🤖 RAG чатбот по игре НордГвард: новые земли.")
st.caption("Задай вопрос — бот ответит на основе твоей базы (Chroma) и покажет источники.")

question = st.text_input("Вопрос", placeholder="Например: Как определяется победитель в игре?")

col1, col2 = st.columns([1, 2])
with col1:
    top_k = st.slider("Top-K chunks", 1, 10, 5)

ask = st.button("Ask 🚀", use_container_width=True)

if ask and question.strip():
    with st.spinner("Думаю..."):
        try:
            resp = requests.post(API_URL, json={"chat_id": "1", "question": question, "top_k": top_k}, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            st.error(f"Ошибка запроса к API: {e}")
            st.stop()

    st.subheader("✅ Ответ")
    st.write(data.get("answer", ""))

    chunks = data.get("chunks", [])
    if chunks:
        st.subheader("📌 Источники (top chunks)")
        for i, c in enumerate(chunks, 1):
            with st.expander(f"Chunk #{i} \n **score:** {c.get('score', 'n/a')}"):
                st.markdown(f"**source:** `{c.get('source', '')}`")
                st.write(c.get("text", ""))