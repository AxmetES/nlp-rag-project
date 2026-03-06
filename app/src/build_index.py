from pathlib import Path
from datetime import datetime

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from config import settings
from src.utils import clean_pdf_docs_inplace


def build_index(force: bool = False):
    db_dir = Path(settings.CHROMA_DIR)

    if force and db_dir.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        db_dir.rename(db_dir.with_name(f"{db_dir.name}__BROKEN__{ts}"))
    db_dir.mkdir(parents=True, exist_ok=True)

    # дальше: load pdf -> clean -> split -> Chroma.from_documents(...)

    # Replace 'your_file.pdf' with the path to your PDF file
    pdf_path = Path(settings.RAW_PDF)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    pdf_loader = PyPDFLoader(str(pdf_path))
    pdf_docs = pdf_loader.load()

    clean_pdf_docs_inplace(pdf_docs)

    #split->
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = text_splitter.split_documents(pdf_docs)

    #embeddings->
    embeddings = OpenAIEmbeddings(model=settings.EMBEDDING_MODEL, openai_api_key=settings.OPENAI_API_KEY)
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=str(settings.CHROMA_DIR),
        collection_name=settings.COLLECTION_NAME)
    print("✅ Built index. Count:", vectorstore._collection.count())
    return vectorstore




