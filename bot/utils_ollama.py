import os
import requests
import numpy as np
from sentence_transformers import SentenceTransformer, util


OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = "bge-m3"                 # re-embed once with this (multilingual)
CHAT_MODEL  = "qwen2.5:3b-instruct-q4_0"


# Load tiny model (downloads ~80MB)
clf_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

INTENT_ANCHORS = {
    "global": [
    "Provide a high-level summary of the entire document.",
    "What are the key takeaways from the whole report?",
    "Give me an overall overview across all sections.",
    "Summarize the document."
    ],
    "section": [
    "Answer a specific question about a detail.",
    "Return a single concrete fact or number.",
    "Focus on a particular section or page."
    ],
}

anchor_embs = {label: clf_model.encode(anchors, convert_to_tensor=True)
               for label, anchors in INTENT_ANCHORS.items()}

def classify_query_small(query: str, threshold: float = 0.35):
    qv = clf_model.encode(query, convert_to_tensor=True)
    best_label, best_score = "unknown", -1.0
    for label, vecs in anchor_embs.items():
        score = util.cos_sim(qv, vecs).max().item()
        if score > best_score:
            best_label, best_score = label, score
    if best_score < threshold:
        return "section", best_score
    return best_label

def build_prompt(question, context, lang_code):
    name = {"en":"English","es":"Spanish","fr":"French"}[lang_code]
    missing = {"en":"Not enough information.",
               "es":"Información insuficiente.",
               "fr":"Informations insuffisantes."}[lang_code]
    return f"""Answer in {name}. Use only the CONTEXT.
            If the answer is missing, say: "{missing}"

            QUESTION:
            {question}

            CONTEXT:
            {context}"""

def local_llm(prompt: str, model: str = CHAT_MODEL) -> str:
    r = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json={
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_ctx": 8192,    # more room for chunks
            "temperature": 0.4  # steadier answers
        }
    }, timeout=180)
    r.raise_for_status()
    return r.json()["response"].strip()



def ollama_embed(texts):
    r = requests.post(
        f"{OLLAMA_BASE_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": texts, "options": {"truncate": True}},
        timeout=120
    )
    r.raise_for_status()
    embs = np.array(r.json()["embeddings"], dtype="float32")
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
    return embs / norms

def ollama_chat(system, user):
    r = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": CHAT_MODEL,
            "stream": False,
            "messages": [
                {"role":"system","content":system},
                {"role":"user","content":user}
            ],
            "options": {"temperature": 0.1, "num_ctx": 1024}
        },
        timeout=180
    )
    r.raise_for_status()
    return r.json()["message"]["content"]
