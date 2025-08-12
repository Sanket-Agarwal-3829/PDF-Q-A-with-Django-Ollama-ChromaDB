import io, re
from collections import Counter
from pypdf import PdfReader
from dataclasses import dataclass

CHUNK_CHARS = 1500
OVERLAP = 500

import argparse, json, pathlib, uuid
from typing import List, Dict, Any

# --- 1) Extract with marker-pdf (Markdown) ---
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

# --- 2) Split with LangChain ---
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


def extract_markdown(pdf_path: str) -> Dict[str, Any]:
    """Return {'markdown': str, 'docinfo': dict} from marker."""
    converter = PdfConverter(create_model_dict())
    md, docinfo, _ = text_from_rendered(converter(pdf_path))
    print('done markdownn')
    return {"markdown": md, "docinfo": docinfo or {}}


def split_markdown(
    markdown_text: str,
    chunk_size: int = 300,
    chunk_overlap: int = 50,
) -> List[Dict[str, Any]]:
    """
    Heading-aware split, then token-aware recursive chunking.
    Returns list of {'id','text','metadata'} dicts.
    """
    # First: split by Markdown headings (preserves section context)
    headers_to_split_on = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]
    mh = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    sections = mh.split_text(markdown_text)  # list of LangChain Documents
    print('done MarkdownHeaderTextSplitter')
    # Second: token-aware recursive splitter
    tok_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # sensible separator order for Markdown/PDF text
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    out = []
    for sec in sections:
        docs = tok_splitter.create_documents(
            [sec.page_content],
            metadatas=[sec.metadata],  # carries h1/h2/h3 etc.
        )
        for d in docs:
            out.append(
                {
                    "id": str(uuid.uuid4()),
                    "text": d.page_content,
                    "metadata": d.metadata.copy(),
                }
            )
    return out



def pdf_toxunks(pdf_path):

    extraction = extract_markdown(pdf_path)
    md = extraction["markdown"]

    chunks = split_markdown(md, chunk_size=300, chunk_overlap=200)
    return chunks
