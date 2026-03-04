from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from app.config import settings
from app.src.utils import docs_to_context

embeddings = OpenAIEmbeddings(
    model=settings.EMBEDDING_MODEL,
    api_key=settings.OPENAI_API_KEY,
)

vectorstore = Chroma(
    collection_name=settings.COLLECTION_NAME,
    persist_directory=str(settings.CHROMA_DIR),
    embedding_function=embeddings,
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})


def rag_context(question: str) -> str:
    try:
        docs = retriever.invoke(question)
        return docs_to_context(docs)
    except Exception as e:
        return f"Error: {e}"