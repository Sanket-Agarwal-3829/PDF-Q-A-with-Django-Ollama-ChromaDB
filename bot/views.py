from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.http import JsonResponse
from django.conf import settings

import json
import os


import re
from collections import OrderedDict
from typing import List, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from pypdf import PdfReader
from collections import Counter
import re
import requests
import numpy as np
from dataclasses import dataclass
import io
from typing import List, Tuple, Dict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import chromadb
from .utils_pdf import pdf_toxunks
from .utils_ollama import ollama_embed, ollama_chat, build_prompt,local_llm, classify_query_small

from numbers import Number

client = chromadb.PersistentClient(path=os.path.join(settings.BASE_DIR, "bot/chroma_store"))
collection = client.get_or_create_collection(name="pdf_chunks", metadata={"hnsw:space": "cosine"})

ALLOWED_SCALARS = (bool, int, float, str)

def to_scalar(v):
    if v is None:
        return None
    # unwrap numpy/torch scalars if you have them
    try:
        import numpy as np  # optional
        if isinstance(v, (np.generic,)):
            v = v.item()
    except Exception:
        pass

    # already a allowed scalar
    if isinstance(v, ALLOWED_SCALARS):
        return v

    # common coercions
    if isinstance(v, Number):
        return float(v)
    if isinstance(v, (list, dict, tuple, set)):
        # stringify complex types (or json.dumps if you prefer)
        try:
            return json.dumps(v, ensure_ascii=False)
        except Exception:
            return str(v)

    # fallback
    return str(v)

def sanitize_metadata(md: dict) -> dict:
    clean = {}
    for k, v in (md or {}).items():
        sv = to_scalar(v)
        if sv is not None:
            clean[k] = sv
    return clean

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()

def _candidates_from_docs(docs: List[str], max_feats: int = 2000, top_terms: int = 30) -> List[str]:
 
    if not docs:
        return []
    vec = TfidfVectorizer(ngram_range=(1,2), stop_words="english",
                          max_features=max_feats, min_df=1)
    X = vec.fit_transform([_norm(d) for d in docs])
    scores = np.asarray(X.sum(axis=0)).ravel()
    feats = vec.get_feature_names_out()
    order = scores.argsort()[::-1][:top_terms]
    terms = [feats[i] for i in order]
    # keep readable terms (letters, numbers, hyphens, spaces)
    terms = [t for t in terms if re.fullmatch(r"[a-z0-9\- ]{3,}", t)]
    return list(OrderedDict.fromkeys(terms))

def _embed(texts, embed_fn) -> np.ndarray:
    v = embed_fn(texts)
    try:
        return np.array(v)
    except Exception:
        return np.array(v.tolist())

def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))

# ---------- main
def retrieve_with_dynamic_rerank(collection, question: str, embed_fn,
                                 k_primary: int = 20, k_final: int = 20) -> dict:

    qv = _embed([question], embed_fn)[0]

    # (1) primary retrieval
    res = collection.query(
        query_embeddings=[qv.tolist()],
        n_results=k_primary,
        include=["documents","metadatas","distances"]
    )
    ids  = (res.get("ids") or [[]])[0] or []
    docs = (res.get("documents") or [[]])[0] or []
    metas= (res.get("metadatas") or [[]])[0] or []
    dists= (res.get("distances") or [[]])[0] or []

    if not ids:
        return res

    cand_terms = _candidates_from_docs(docs, top_terms=40)

    if cand_terms:
        term_vecs = _embed(cand_terms, embed_fn)
        sims = [ _cos(qv, tv) for tv in term_vecs ]
        order = np.argsort(sims)[::-1][:10]
        anchors = [cand_terms[i] for i in order]
    else:
        anchors = []

    # (4) soft rerank
    def bonus(text: str, md: dict) -> float:
        txt = _norm(text)
        head = _norm(" ".join(str(md.get(k,"")) for k in ("h1","h2","h3","title","section")))
        b = 0.0
        for a in anchors:
            if a in txt:  b -= 0.08
            if a in head: b -= 0.05
        return b

    items = list(zip(ids, docs, metas, dists))
    items.sort(key=lambda x: ((x[3] if x[3] is not None else 1.0) + bonus(x[1], x[2])))

    # take final top-k and rebuild Chroma-like response
    items = items[:k_final]
    out_ids  = [x[0] for x in items]
    out_docs = [x[1] for x in items]
    out_meta = [x[2] for x in items]
    out_dist = [x[3] for x in items]

    return {"ids":[out_ids], "documents":[out_docs], "metadatas":[out_meta], "distances":[out_dist]}

def build_context_chunks(res, max_chars=1000):

    context_chunks = []

    if not res or not res.get("ids") or not res["ids"] or not res["ids"][0]:
        return context_chunks

    qidx = 0  # single-query case
    ids_q = res["ids"][qidx]
    docs_q = res.get("documents", [[]])[qidx] or []
    metas_q = res.get("metadatas", [[]])[qidx] or []
    dists_q = (res.get("distances", [[]])[qidx] or [None] * len(ids_q))

    n = min(len(ids_q), len(docs_q), len(metas_q), len(dists_q))

    for i in range(n):
        t = docs_q[i] or ""
        if len(t) > max_chars:
            t = t[:max_chars] + "…"

        md = metas_q[i] or {}
        page = md.get("page") or md.get("page_start") or md.get("page_number") or "?"
        did = ids_q[i]
        dist = dists_q[i]

        prefix = f"[Chunk {i+1} | id:{did}"
        if page != "?":
            prefix += f" | p.{page}"
        if dist is not None:
            prefix += f" | score:{dist:.4f}"
        prefix += "]"

        context_chunks.append(f"{t}")

    return context_chunks



def index(request):
    pdf_dir = os.path.join(settings.MEDIA_ROOT, "pdfs")
    pdf_list = []

    if os.path.exists(pdf_dir):
        for file_name in os.listdir(pdf_dir):
            if file_name.lower().endswith(".pdf"):
                pdf_list.append({
                    "name": file_name,
                    "url": settings.MEDIA_URL + "pdfs/" + file_name
                })

    return render(request, "main.html", {"pdf_list": pdf_list})


def upload_pdf(request):
    if request.method == "POST" and request.FILES.get("pdf"):
        pdf_file = request.FILES["pdf"]
        pdf_path = os.path.join(settings.MEDIA_ROOT, "pdfs", pdf_file.name)
        os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
        with open(pdf_path, "wb") as f:
            for chunk in pdf_file.chunks():
                f.write(chunk)
        
        chunks = pdf_toxunks(pdf_path)

        texts = [c["text"] for c in chunks]
        embs = ollama_embed(texts)
        emb_list = embs if isinstance(embs, list) else embs.tolist()

        ids = [c.get("id") or f"doc1-{i}" for i, c in enumerate(chunks)]

        metas = []
        for c in chunks:
            md = c.get("metadata") or {}
            base = {
                "doc_id": md.get("doc_id") or md.get("source") or "doc1",
                "page": md.get("page") or md.get("page_start") or md.get("page_number"),
                "offset": md.get("offset"),
            }
            # merge with the rest, then sanitize
            merged = {**md, **base}
            metas.append(sanitize_metadata(merged))

        # (optional) sanity check lengths
        assert len(ids) == len(texts) == len(metas) == len(emb_list)

        collection.add(
            ids=ids,
            embeddings=emb_list,
            documents=texts,
            metadatas=metas,
        )


    return redirect('index')


def ask_question(request):
    answer = None
    if request.method == "POST":
        data = json.loads(request.body)
        q = data.get("question", "")
        lang_code = data.get("lang_code", "")
        classification = classify_query_small(q)
        k = 30 if classification == "global" else 20

        res = retrieve_with_dynamic_rerank(collection, q, embed_fn=ollama_embed,
                                        k_primary=k, k_final=min(k, 20))

        context_chunks = build_context_chunks(res, max_chars=1000)  # your safe formatter

        print(len(context_chunks))

        context_text = "\n\n".join(context_chunks)
        prompt = build_prompt(q, context_text, lang_code)
        print(prompt)
        answer = local_llm(prompt)

    return JsonResponse({"answer": f"You asked: {answer}", 'context_chunks':"\n\n".join(context_chunks)})
