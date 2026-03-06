from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from config import settings

def get_vectorstore() -> Chroma:
    embeddings = OpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        api_key=settings.OPENAI_API_KEY,
    )
    return Chroma(
        collection_name=settings.COLLECTION_NAME,
        persist_directory=str(settings.CHROMA_DIR),
        embedding_function=embeddings,
    )


def is_index_ready(vs: Chroma) -> bool:
    try:
        return vs._collection.count() > 0
    except Exception:
        # битая/несовместимая база тоже сюда попадёт
        return False


def rag_context(question: str, vs: Chroma, k: settings.TOP_K) -> list:
    # retriever = vs.as_retriever(search_kwargs={"k": k})
    # docs = retriever.invoke(question)
    return vs.similarity_search_with_score(question, k=k)
    # return docs