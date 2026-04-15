"""
rag.py
FAISS-based Retrieval Augmented Generation for MarketMind.

Builds one vector index per ticker from news headlines.
At query time, embeds a natural language query and retrieves
the k most semantically relevant headlines — not date-filtered.

This is genuine RAG: irrelevant noise is filtered automatically
and only contextually relevant headlines reach the prediction prompt.
"""

import numpy as np
import pandas as pd
from typing import List

_model = None
_faiss = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print("  Loading sentence embedding model (~30 seconds first time)...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        print("  Embedding model ready.")
    return _model


def _get_faiss():
    global _faiss
    if _faiss is None:
        import faiss
        _faiss = faiss
    return _faiss


class NewsRAG:
    def __init__(self):
        self.indices = {}
        self.texts   = {}
        self.built   = False

    def build(self, news_df: pd.DataFrame):
        if news_df is None or news_df.empty:
            print("  WARNING: No news data to index.")
            self.built = False
            return

        model = _get_model()
        faiss = _get_faiss()

        for ticker in news_df["Ticker"].unique():
            headlines = (news_df[news_df["Ticker"] == ticker]["Title"]
                         .dropna().tolist())
            headlines = [h for h in headlines if h and str(h) != "nan"]
            if not headlines:
                continue

            embeddings = model.encode(headlines, convert_to_numpy=True,
                                      show_progress_bar=False)
            norms      = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms      = np.where(norms == 0, 1, norms)
            embeddings = (embeddings / norms).astype("float32")

            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)
            self.indices[ticker] = index
            self.texts[ticker]   = headlines

        self.built = True
        total = sum(len(v) for v in self.texts.values())
        print(f"  FAISS index built: {len(self.indices)} tickers, {total} headlines")

    def retrieve(self, ticker: str, query: str, k: int = 5) -> List[str]:
        if not self.built or ticker not in self.indices:
            return self.texts.get(ticker, [])[:k]

        model = _get_model()
        q_emb = model.encode([query], convert_to_numpy=True)
        norm  = np.linalg.norm(q_emb)
        if norm > 0:
            q_emb = q_emb / norm

        _, idxs   = self.indices[ticker].search(q_emb.astype("float32"), k)
        headlines = self.texts[ticker]
        return [headlines[i] for i in idxs[0] if 0 <= i < len(headlines)]


def build_rag_from_df(news_df: pd.DataFrame) -> NewsRAG:
    rag = NewsRAG()
    rag.build(news_df)
    return rag
