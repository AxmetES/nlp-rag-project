import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = BASE_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH)

class Settings:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    def validate(self) -> None:
        if not self.OPENAI_API_KEY:
            raise RuntimeError(f"OPENAI_API_KEY не найден. Проверь {ENV_PATH}")


    APP_DIR = Path(__file__).resolve().parents[0]
    DATA_JSONL = APP_DIR / "data" / "processed" / "docs.jsonl"
    CHROMA_DIR = APP_DIR / "data" / "index" / "chroma"
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    COLLECTION_NAME = "nordgard_rules"

    RAW_PDF = APP_DIR / "data" / "raw" / "nordguard_game_tutorial.pdf"

    LLM_MODEL="gpt-4o-mini"
    EMBEDDING_MODEL="text-embedding-3-small"
    TOP_K = 5

settings = Settings()
settings.validate()