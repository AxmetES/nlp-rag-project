import re
import json
from pathlib import Path

from schemas import Chunk


def save_splits_to_jsonl(splits, out_path: str):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for i, d in enumerate(splits):
            rec = {
                "id": f"{d.metadata.get('source','doc')}:{d.metadata.get('page', 'na')}:{i}",
                "text": d.page_content,
                "meta": d.metadata,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"💾 Saved {len(splits)} chunks to {out_path}")


def clean_pdf_docs_inplace(docs) -> None:
    for d in docs:
        t = d.page_content or ""

        t = t.replace("\xa0", " ")
        t = t.replace("\u00ad", "")
        t = re.sub(r"(\w)[ \t]*-\n[ \t]*(\w)", r"\1\2", t)
        t = t.replace("\n\n", "<PARA>")
        t = t.replace("\n", " ")
        t = t.replace("<PARA>", "\n\n")
        t = re.sub(r"(?m)^\s*z\s*$", "", t)
        t = re.sub(r"(?<=\s)z(?=\s)", " ", t)
        t = re.sub(r"[ \t]{2,}", " ", t)
        t = re.sub(r"\n{3,}", "\n\n", t)

        d.page_content = t.strip()


from typing import List, Any
from sentence_transformers import CrossEncoder

_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

reranker = CrossEncoder(_MODEL_NAME)

def rerank(query: str, docs: List, top_k: int = 5) -> List:
    pairs = [(query, d.page_content) for d in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked[:top_k]]


def docs_to_context(docs, max_docs=5):
    parts = []
    for d in docs[:max_docs]:
        page = d.metadata.get("page")
        parts.append(f"[стр.{page}] {d.page_content.strip()}")
    return "\n\n".join(parts)


def docs_to_chunks(docs_scores) -> list[Chunk]:
    out: list[Chunk] = []
    for doc, score in docs_scores:
        meta = doc.metadata or {}
        source = meta.get("source") or meta.get("file") or meta.get("path") or meta.get("page_label")
        out.append(Chunk(
            text=doc.page_content,
            source=str(source) if source is not None else None,
            score=float(score) if score is not None else None,
        ))
    return out