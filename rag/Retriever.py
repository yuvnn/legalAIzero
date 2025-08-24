
import os
from dotenv import load_dotenv
import json, numpy as np

import torch
import faiss
from sentence_transformers import SentenceTransformer

from config import EMB_FILE

with open(EMB_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

texts = [d.get("text","") for d in data if isinstance(d.get("text"), str)]
embs  = np.asarray([d["embedding"] for d in data if isinstance(d.get("embedding"), list)], dtype="float32")
meta  = [{
    "doc_id": d.get("doc_id"),
    "chunk_index": d.get("chunk_index"),
    "filename": d.get("filename"),
    "folder": d.get("folder"),
} for d in data]

def _normalize(mat):
    n = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / n

dim = embs.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(_normalize(embs).astype("float32"))

def dense_search_by_vec(query_vec, k=5):
    """쿼리 벡터(query_vec)로 FAISS 인덱스에서 top-k 유사 문서 검색"""
    q = _normalize(query_vec[None, :].astype("float32"))
    D, I = index.search(q, k)
    return [(int(i), float(s)) for i, s in zip(I[0], D[0])]

def e5_query_embed(q: str):
    """질문(q)을 E5 모델로 임베딩 벡터로 변환"""
    global _e5
    if '_e5' not in globals() or _e5 is None:
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        _e5 = SentenceTransformer("intfloat/multilingual-e5-large-instruct", device=DEVICE)
        _e5.max_seq_length = 512
    qf = "query: " + q.strip()
    v = _e5.encode([qf], normalize_embeddings=True, show_progress_bar=False)[0]
    return v.astype("float32")

def retriever_dense(query, k=5):
    """질문(query)로 top-k 문서 검색 결과 반환"""
    v = e5_query_embed(query)
    hits = dense_search_by_vec(v, k=k)
    docs = [{
        "doc_id": meta[i].get("doc_id"),
        "chunk_index": meta[i].get("chunk_index"),
        "score": score,
        "filename": meta[i].get("filename"),
        "text": texts[i]
    } for (i,score) in hits]
    return {"retrieved_docs": docs}

if __name__ == "__main__":
    query = "종합부동산세법의 목적은 무엇인가요?"
    result = retriever_dense(query, k=5)
    print("검색 결과:")
    for doc in result["retrieved_docs"]:
        print(f"doc_id: {doc['doc_id']}, score: {doc['score']:.4f}, text: {doc['text'][:80]}...")