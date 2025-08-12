# PDF Q\&A with Django + Ollama + ChromaDB

Ask questions about your PDFs in natural language. This app:

* Extracts **Markdown** from PDFs with `marker-pdf`
* Splits & indexes chunks into a **ChromaDB** vector store
* Uses **Ollama** for both **embeddings** (`bge-m3`) and **chat** (`qwen2.5:3b-instruct-q4_0`)
* Classifies query “intent” (global summary vs. section/detail) for smarter retrieval
* Answers strictly from retrieved context (multi-lingual: English, Spanish, French)

---

## ✨ Features

* **PDF → Markdown** via `marker-pdf` for higher-quality chunks
* **Heading-aware** + **token-aware** chunking (LangChain splitters)
* **Dynamic re-ranking** with TF-IDF anchor terms for better top-k
* **Ollama-native** embeddings + chat (fully local)
* **Multilingual answers** (`en`, `es`, `fr`) with graceful “not enough info” handling
* Simple **Django** UI for upload + ask, plus JSON APIs

---

## 📦 Tech Stack

* **Backend:** Django
* **Vector DB:** ChromaDB (persistent local store)
* **LLM & Embeddings:** Ollama (`bge-m3`, `qwen2.5:3b-instruct-q4_0`)
* **Extraction:** `marker-pdf`
* **NLP utils:** `sentence-transformers`, `scikit-learn`, `langchain`, `pypdf`

---

## 🚀 Quickstart

### 1) Prerequisites

* Python 3.10+ (recommended)
* [Ollama](https://ollama.com/) installed and running locally
* `git`, `virtualenv` (optional but recommended)

### 2) Pull models in Ollama

```bash
ollama pull bge-m3
ollama pull qwen2.5:3b-instruct-q4_0
```

> If you prefer other models, see **Config** below.

### 3) Clone & set up environment

```bash
git clone https://github.com/<you>/<repo>.git
cd <repo>
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**requirements.txt** (you can copy this):

```
Django>=5.0
chromadb>=0.5.0
numpy>=1.26
requests>=2.31
pypdf>=4.0
sentence-transformers>=3.0
scikit-learn>=1.4
langchain>=0.2
tiktoken>=0.7
marker-pdf>=0.2
```

> `marker-pdf` may install PyTorch; ensure your platform has compatible wheels. If install fails, try upgrading `pip`, `setuptools`, `wheel`.

### 4) Environment variables

Create `.env` (or export in your shell):

```bash
# Ollama host (defaults to http://localhost:11434)
export OLLAMA_BASE_URL="http://localhost:11434"

# Django settings (optional example)
export DJANGO_DEBUG=1
```

If you use `python-dotenv`, load `.env` in `manage.py` or `settings.py`. Otherwise export in your shell profile.

### 5) Django settings (media & Chroma path)

In `settings.py`:

```python
import os
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent

MEDIA_URL = "/media/"
MEDIA_ROOT = BASE_DIR / "media"

# Templates: ensure 'DIRS' has your templates path where main.html lives
TEMPLATES[0]["DIRS"] = [BASE_DIR / "templates"]
```

This project stores the vector DB at: `bot/chroma_store` (relative to `BASE_DIR`).

### 6) URL routes

`urls.py`:

```python
from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from bot import views  # adjust app name if different

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", views.index, name="index"),
    path("upload_pdf/", views.upload_pdf, name="upload_pdf"),
    path("ask_question/", views.ask_question, name="ask_question"),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
```

### 7) Run the app

Start **Ollama** first:

```bash
ollama serve
```

Then run Django:

```bash
python manage.py migrate
python manage.py runserver
```

Open: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)

Upload a PDF, ask a question, and get an answer with cited context (server-side).

---

## 🗂️ Project Structure

```
repo/
├─ bot/
│  ├─ views.py               # index, upload_pdf, ask_question
│  ├─ utils_pdf.py           # pdf → markdown → chunks (see code below)
│  ├─ utils_ollama.py        # ollama_embed, ollama_chat, local_llm, build_prompt, classify
│  ├─ chroma_store/          # ChromaDB persistent store (auto-created)
│  └─ templates/
│     └─ main.html           # minimal UI (upload + ask)
├─ media/
│  └─ pdfs/                  # uploaded PDFs
├─ manage.py
├─ requirements.txt
└─ README.md
```

> In your provided code, the PDF utility function is named `pdf_toxunks` (note the “x”); keep that name or rename consistently.

---

## 🧠 Core Logic (how it works)

1. **Extract Markdown**

   ```python
   from marker.converters.pdf import PdfConverter
   from marker.models import create_model_dict
   from marker.output import text_from_rendered
   ```

   Converts PDF → rich Markdown + doc metadata.

2. **Split into chunks**

   * First pass: `MarkdownHeaderTextSplitter` to preserve section hierarchy (`h1/h2/h3`).
   * Second pass: `RecursiveCharacterTextSplitter.from_tiktoken_encoder` into token-aware chunks.

3. **Embed & Index**

   * `ollama /api/embed` with `bge-m3`
   * Store `ids`, `documents`, `metadatas`, `embeddings` in **Chroma** (cosine space).

4. **Query Time**

   * Classify query intent (`global` vs `section`) with a tiny **SentenceTransformer** to adjust `k`.
   * Retrieve `k_primary`, derive **anchor terms** via TF-IDF on candidates, then **soft re-rank** by boosting matches in text/headers.
   * Build context (truncate per chunk), then call **local LLM** via `ollama /api/generate` with a strict system prompt: “Use only the CONTEXT.”

---

## 🔌 API Endpoints

### Upload PDF

`POST /upload_pdf/`
`multipart/form-data` with field `pdf`

**curl**

```bash
curl -F "pdf=@/path/to/file.pdf" http://127.0.0.1:8000/upload_pdf/
```

### Ask a Question

`POST /ask_question/`
`Content-Type: application/json`

Body:

```json
{
  "question": "What are the key findings?",
  "lang_code": "en"
}
```

Response:

```json
{
  "answer": "You asked: ...", 
  "context_chunks": "..."
}
```

> The view wraps the raw model output as `"You asked: {answer}"`. Modify to suit your UI.

---

## 🧩 Minimal Frontend Example (`templates/main.html`)

Your simplest form can:

* Show existing PDFs
* Upload a new PDF
* Ask a question with CSRF token

> If you use Django’s CSRF middleware, include the token. Or mark `ask_question` view `@csrf_exempt` for quick testing (not recommended for prod).

```html
<form action="/upload_pdf/" method="post" enctype="multipart/form-data">
  {% csrf_token %}
  <input type="file" name="pdf" accept="application/pdf" required />
  <button type="submit">Upload PDF</button>
</form>

<input id="q" placeholder="Ask a question" />
<select id="lang">
  <option value="en">English</option>
  <option value="es">Spanish</option>
  <option value="fr">French</option>
</select>
<button id="ask">Ask</button>

<pre id="answer"></pre>
<pre id="ctx"></pre>

<script>
async function getCSRF() {
  return document.cookie.split('; ').find(r => r.startsWith('csrftoken='))?.split('=')[1];
}
document.getElementById('ask').onclick = async () => {
  const csrftoken = await getCSRF();
  const res = await fetch('/ask_question/', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'X-CSRFToken': csrftoken || '' },
    body: JSON.stringify({
      question: document.getElementById('q').value,
      lang_code: document.getElementById('lang').value
    })
  });
  const data = await res.json();
  document.getElementById('answer').textContent = data.answer;
  document.getElementById('ctx').textContent = data.context_chunks;
};
</script>
```

---

## ⚙️ Configuration

The main toggles live in `utils_ollama.py` (or the module where you placed them):

```python
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = "bge-m3"                 # multilingual embeddings
CHAT_MODEL  = "qwen2.5:3b-instruct-q4_0"

# Chat defaults
options = {
  "num_ctx": 8192,  # raise if you need longer context (watch VRAM)
  "temperature": 0.4
}
```

**Swap models** (examples):

* `CHAT_MODEL="llama3.1:8b-instruct-q4_K_M"`
* `EMBED_MODEL="nomic-embed-text"` (also multilingual options exist)

Update your `ollama pull <model>` accordingly.

---

## 🛡️ Notes on Safety & Grounding

* The prompt enforces “**Use only the CONTEXT**. If missing, reply with ‘Not enough information.’”
* Retrieval is cosine similarity + soft re-rank; adjust thresholds if you get hallucinations.
* Consider returning **source chunk ids/pages** in the UI for transparency.

---

## 🧪 Troubleshooting

* **`marker-pdf` install issues**
  Upgrade build tools:
  `pip install --upgrade pip setuptools wheel`
  On some platforms, installing/upgrade **PyTorch** first helps.

* **Ollama connection errors**
  Ensure `ollama serve` is running and `OLLAMA_BASE_URL` matches.
  Test: `curl http://localhost:11434/api/tags`

* **Memory / VRAM**
  If you hit OOM, drop to a smaller chat model or reduce `num_ctx`.

* **Empty results from Chroma**
  Check that `upload_pdf` actually added documents and that the collection path is writeable.

* **CSRF 403 on `/ask_question/`**
  Include CSRF token (see example) or add `@csrf_exempt` during development.

---

## 🧠 Intent Classification

Tiny classifier (`all-MiniLM-L6-v2`) picks between:

* `global`: “summarize the whole document”
* `section`: “answer a specific detail”

Default threshold returns `section` if uncertain. Tweak `threshold` in `classify_query_small`.

---

## 🔒 Production Tips

* Add **chunk/page IDs** to the final answer as citations
* Cache embeddings & skip re-embedding existing PDFs
* Add auth/rate limits
* Enable gzip & `SECURE_*` Django settings
* Containerize Ollama + app with GPU runtime (optional)

---

## 🧰 Dev Helpers

**Rebuild the Chroma store** (delete and re-upload PDFs) if schema changes:

```bash
rm -rf bot/chroma_store
```

---

## ✅ Example JSON Response

```json
{
  "answer": "You asked: The report concludes that ...",
  "context_chunks": "[Chunk 1 | id:... | p.12 | score:0.0931] ...\n\n[Chunk 2 | id:... | p.13 | score:0.1022] ..."
}
```

---

## 📄 License

MIT — feel free to use, modify, and contribute.

---

## 🙌 Acknowledgements

* [`marker-pdf`](https://github.com/chrismattmann/marker)
* [`chromadb`](https://www.trychroma.com/)
* [`sentence-transformers`](https://www.sbert.net/)
* [`LangChain`](https://python.langchain.com/)
* [`Ollama`](https://ollama.com/)

---

## ✍️ Credits

Built with ♥ using Django, local LLMs, and a sprinkle of TF-IDF re-ranking. If you use this, star the repo and open an issue/PR with ideas!
