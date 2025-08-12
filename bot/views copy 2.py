from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.http import JsonResponse
from django.conf import settings

import json
import os

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
from .utils_pdf import *
from .utils_ollama import ollama_embed, ollama_chat, build_prompt,local_llm, classify_query_small

client = chromadb.PersistentClient(path=os.path.join(settings.BASE_DIR, "bot/chroma_store"))
collection = client.get_or_create_collection(name="pdf_chunks", metadata={"hnsw:space": "cosine"})


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
        
        pages = read_pdf_pages(pdf_path)          # List[str]
        chunks = make_chunks(pages)  
        print(chunks) 
        texts = [c["text"] for c in chunks]
        embs = ollama_embed(texts)
        emb_list = embs if isinstance(embs, list) else embs.tolist()

        ids = [c.get("id") or f"doc1-{i}" for i, c in enumerate(chunks)]
        metas = [
            {
                "page": c.page,
                "offset": c.offset,
                "source": pdf_path,
                "filename": pdf_file.name,
            }
            for c in chunks
        ]

        # 5) embeddings
        embs = ollama_embed(texts)
        emb_list = embs if isinstance(embs, list) else embs.tolist()

        # sanity check
        assert len(ids) == len(texts) == len(metas) == len(emb_list)

        # 6) write to vector store
        collection.add(
            ids=ids,
            embeddings=emb_list,
            documents=texts,
            metadatas=metas,
        )


    return redirect('index')


# usage





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

        # for i in range(len(res["ids"][0])):
        #     t = res["documents"][0][i]
        #     if len(t) > 1000:
        #         t = t[:1000] + "…"
        #     page = res["metadatas"][0][i]["page"]
        #     context_chunks.append(f"[Chunk {i+1} | p.{page}] {t}")
        print(len(context_chunks))

        context_text = "\n\n".join(context_chunks)
        prompt = build_prompt(q, context_text, lang_code)
        print(prompt)
        answer = local_llm(prompt)

    return JsonResponse({"answer": f"You asked: {answer}", 'context_chunks':"\n\n".join(context_chunks)})
