from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from config import settings

load_dotenv()

llm = ChatOpenAI(
    model=settings.LLM_MODEL,
    temperature=0.2,
    api_key=settings.OPENAI_API_KEY,
)


def run_llm(context: str, question: str, history_text: str = "") -> str:
    # Prompt
    template = """Представь что ты эксперт по настольной играе НОРДГАРД: НОВЫЕ ЗЕМЛИ, отвечай строго по контексту. 
    Ничего не додумывай.
    Если в контексте есть разные режимы (например, игра командами), явно раздели ответ:
    - Обычная игра (каждый сам за себя)
    - Игра командами (если описано).

    ИСТОРИЯ: 
    {history}
    
    Контекст:
    {context}

    Вопрос: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    chain = prompt | llm
    result = chain.invoke({"history": history_text, "context": context, "question": question})

    return result.content