from pathlib import Path
import zipfile, textwrap

base = Path("/mnt/data/ai_document_assistant")
base.mkdir(exist_ok=True)

app_py = r'''import io
import os
import re
import shutil
import tempfile
from pathlib import Path

import faiss
import gdown
import numpy as np
import streamlit as st
from docx import Document
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="AI Document Assistant", page_icon="📄", layout="wide")
st.title("📄 AI Document Assistant")
st.caption("PDF • DOCX • TXT • MD | Hybrid search • FAISS • Groq")


# -----------------------------
# Cached resources
# -----------------------------
@st.cache_resource
def load_embedding_model():
    # Small, fast, general-purpose sentence-transformer.
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def load_faiss_index(embeddings_bytes):
    """Create a FAISS index from already-created embeddings."""
    embeddings = np.frombuffer(embeddings_bytes, dtype="float32")
    dimension = embeddings.shape[1]
    embeddings = embeddings.reshape(-1, dimension)
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index


# -----------------------------
# Document extraction
# -----------------------------
def extract_pdf(file_bytes, filename):
    reader = PdfReader(io.BytesIO(file_bytes))
    records = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            records.append(
                {"text": text.strip(), "filename": filename, "page": page_number}
            )

    return records


def extract_docx(file_bytes, filename):
    document = Document(io.BytesIO(file_bytes))
    paragraphs = [p.text.strip() for p in document.paragraphs if p.text.strip()]
    text = "\n".join(paragraphs)

    if not text:
        return []

    return [{"text": text, "filename": filename, "page": None}]


def extract_txt(file_bytes, filename):
    text = file_bytes.decode("utf-8", errors="ignore").strip()
    if not text:
        return []

    return [{"text": text, "filename": filename, "page": None}]


def extract_md(file_bytes, filename):
    text = file_bytes.decode("utf-8", errors="ignore").strip()
    if not text:
        return []

    return [{"text": text, "filename": filename, "page": None}]


def extract_document(file_bytes, filename):
    extension = Path(filename).suffix.lower()

    if extension == ".pdf":
        return extract_pdf(file_bytes, filename)
    if extension == ".docx":
        return extract_docx(file_bytes, filename)
    if extension == ".txt":
        return extract_txt(file_bytes, filename)
    if extension == ".md":
        return extract_md(file_bytes, filename)

    return []


# -----------------------------
# Chunking
# -----------------------------
def chunk_text(records, chunk_size=800, overlap=120):
    """Create overlapping chunks while preserving filename/page metadata."""
    chunks = []

    for record in records:
        text = re.sub(r"\s+", " ", record["text"]).strip()

        if not text:
            continue

        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end].strip()

            if chunk:
                chunks.append(
                    {
                        "text": chunk,
                        "filename": record["filename"],
                        "page": record["page"],
                    }
                )

            if end >= len(text):
                break

            start = end - overlap

    return chunks


# -----------------------------
# Embeddings + FAISS
# -----------------------------
def create_embeddings(chunks, model):
    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype("float32")

    faiss.normalize_L2(embeddings)
    return embeddings


def build_document_store(chunks, model):
    embeddings = create_embeddings(chunks, model)

    # Store embeddings and metadata in session state so they survive reruns.
    return {
        "chunks": chunks,
        "embeddings": embeddings,
        "embedding_bytes": embeddings.tobytes(),
    }


# -----------------------------
# Keyword search
# -----------------------------
STOP_WORDS = {
    "the", "and", "for", "with", "from", "this", "that", "what", "when",
    "where", "which", "who", "how", "why", "are", "was", "were", "is",
    "to", "of", "in", "on", "a", "an", "as", "by", "or", "it", "be",
    "about", "can", "does", "do", "i", "me", "my", "please", "tell"
}


def important_words(question):
    words = re.findall(r"\b[a-zA-Z0-9][a-zA-Z0-9_-]*\b", question.lower())
    return [word for word in words if word not in STOP_WORDS and len(word) > 1]


def keyword_score(question, text):
    words = important_words(question)
    if not words:
        return 0.0

    text_lower = text.lower()
    matches = sum(1 for word in words if word in text_lower)
    return matches / len(words)


# -----------------------------
# Hybrid search
# -----------------------------
def hybrid_search(question, store, index, model, top_k=5):
    """Combine semantic similarity and keyword overlap."""
    question_embedding = model.encode(
        [question], convert_to_numpy=True
    ).astype("float32")
    faiss.normalize_L2(question_embedding)

    # Retrieve more candidates first, then combine scores.
    candidate_k = min(max(top_k * 4, 10), len(store["chunks"]))
    semantic_scores, indices = index.search(question_embedding, candidate_k)

    candidates = []

    for semantic_score, idx in zip(semantic_scores[0], indices[0]):
        if idx < 0:
            continue

        chunk = store["chunks"][idx]
        kw_score = keyword_score(question, chunk["text"])

        # 70% semantic + 30% keyword.
        combined_score = (0.70 * float(semantic_score)) + (0.30 * kw_score)

        candidates.append(
            {
                "text": chunk["text"],
                "filename": chunk["filename"],
                "page": chunk["page"],
                "semantic_score": float(semantic_score),
                "keyword_score": float(kw_score),
                "score": combined_score,
            }
        )

    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates[:top_k]


# -----------------------------
# Google Drive loader
# -----------------------------
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


def load_from_google_drive(url):
    """
    Works with publicly accessible Google Drive file/folder links.
    For private Drive files, use a public/shareable link or a Drive API
    integration with OAuth/service-account credentials.
    """
    temp_dir = tempfile.mkdtemp(prefix="drive_docs_")

    try:
        # gdown can download a public file or recursively download a
        # publicly accessible folder.
        result = gdown.download_folder(
            url,
            output=temp_dir,
            quiet=True,
            use_cookies=False,
        )

        # Some gdown versions return None for a single file/failure.
        files = []

        if result:
            for item in result:
                item_path = Path(item)
                if item_path.is_file() and item_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    files.append(item_path)

        # Fallback: try the URL as a single file.
        if not files:
            guessed_name = "drive_file"
            output_file = Path(temp_dir) / guessed_name
            downloaded = gdown.download(
                url,
                output=str(output_file),
                quiet=True,
                fuzzy=True,
            )

            if downloaded:
                downloaded_path = Path(downloaded)
                if downloaded_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    files.append(downloaded_path)

        loaded = []
        for path in files:
            loaded.append((path.name, path.read_bytes()))

        return loaded

    finally:
        # Files have already been read into memory.
        shutil.rmtree(temp_dir, ignore_errors=True)


# -----------------------------
# Process documents once
# -----------------------------
def process_documents(document_items, model):
    all_records = []
    document_info = []

    for filename, file_bytes in document_items:
        records = extract_document(file_bytes, filename)
        all_records.extend(records)

        document_info.append(
            {
                "filename": filename,
                "type": Path(filename).suffix.lower().replace(".", "").upper(),
                "pages": len(records) if Path(filename).suffix.lower() == ".pdf" else None,
                "characters": sum(len(record["text"]) for record in records),
            }
        )

    chunks = chunk_text(all_records)

    if not chunks:
        return None, document_info

    store = build_document_store(chunks, model)
    return store, document_info


# -----------------------------
# Session state
# -----------------------------
if "document_store" not in st.session_state:
    st.session_state.document_store = None

if "document_info" not in st.session_state:
    st.session_state.document_info = []

if "source_signature" not in st.session_state:
    st.session_state.source_signature = None


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("📚 Add documents")

    uploaded_files = st.file_uploader(
        "Upload files",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
    )

    st.divider()

    st.subheader("Google Drive")
    drive_url = st.text_input(
        "Paste a public Drive file or folder link",
        placeholder="https://drive.google.com/...",
    )

    st.caption(
        "Drive files/folders must be accessible to the link. "
        "Supported: PDF, DOCX, TXT, MD."
    )

    chunk_size = st.slider("Chunk size", 400, 1500, 800, 50)
    overlap = st.slider("Chunk overlap", 50, 300, 120, 10)
    top_k = st.slider("Sources per answer", 1, 10, 5)

    process_button = st.button("⚙️ Process documents", use_container_width=True)


# -----------------------------
# Process button
# -----------------------------
if process_button:
    document_items = []

    if uploaded_files:
        document_items.extend(
            [(file.name, file.getvalue()) for file in uploaded_files]
        )

    if drive_url.strip():
        with st.spinner("Loading Google Drive files..."):
            try:
                drive_items = load_from_google_drive(drive_url.strip())
                document_items.extend(drive_items)

                if not drive_items:
                    st.warning(
                        "No supported files were found. Check that the Drive "
                        "link is public and contains PDF, DOCX, TXT or MD files."
                    )
            except Exception as exc:
                st.error(f"Google Drive loading failed: {exc}")

    if not document_items:
        st.warning("Upload at least one document or provide a Drive link.")
    else:
        model = load_embedding_model()

        with st.spinner("Extracting text, creating chunks and embeddings..."):
            store, info = process_documents(document_items, model)

        if store is None:
            st.error("No text could be extracted from the supplied documents.")
        else:
            # Save the requested chunk settings with the processed store.
            store["chunk_size"] = chunk_size
            store["overlap"] = overlap

            st.session_state.document_store = store
            st.session_state.document_info = info
            st.session_state.source_signature = (
                tuple((name, len(data)) for name, data in document_items),
                chunk_size,
                overlap,
            )

            st.success(
                f"Processed {len(info)} document(s) and created "
                f"{len(store['chunks'])} chunks."
            )


# -----------------------------
# Document information
# -----------------------------
if st.session_state.document_store:
    st.header("📋 Extracted document information")

    info = st.session_state.document_info
    for item in info:
        page_text = (
            f"{item['pages']} PDF pages"
            if item["pages"] is not None
            else "page number not available"
        )
        st.write(
            f"**{item['filename']}** — {item['type']} — "
            f"{item['characters']:,} extracted characters — {page_text}"
        )

    st.info(
        f"Total chunks: **{len(st.session_state.document_store['chunks'])}**. "
        "Embeddings are created once when documents are processed."
    )


# -----------------------------
# Ask question
# -----------------------------
st.header("💬 Ask your documents")

question = st.text_input(
    "Question",
    placeholder="Ask something about the uploaded documents...",
)

if question:
    if not st.session_state.document_store:
        st.warning("Please process documents first.")
    else:
        store = st.session_state.document_store
        model = load_embedding_model()

        # Rebuild the small FAISS index from stored embeddings only.
        # No document embeddings are recalculated.
        index = load_faiss_index(store["embedding_bytes"])

        with st.spinner("Searching documents..."):
            results = hybrid_search(
                question,
                store,
                index,
                model,
                top_k=top_k,
            )

        context_parts = []
        for number, result in enumerate(results, start=1):
            page = (
                f"page {result['page']}"
                if result["page"] is not None
                else "page unavailable"
            )
            context_parts.append(
                f"[Source {number}: {result['filename']}, {page}]\n"
                f"{result['text']}"
            )

        context = "\n\n".join(context_parts)

        api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

        if not api_key:
            st.error(
                "GROQ_API_KEY is missing. Add it to Streamlit Secrets as "
                "`GROQ_API_KEY`."
            )
        else:
            client = Groq(api_key=api_key)

            prompt = f"""
Answer the user's question using ONLY the context below.

Rules:
- Do not use outside knowledge.
- If the answer is not contained in the context, say:
  "The information is not available in the provided documents."
- Do not invent facts, citations, page numbers, or sources.
- Keep the answer clear and concise.

CONTEXT:
{context}

USER QUESTION:
{question}
"""

            with st.spinner("Generating answer..."):
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a document question-answering assistant. "
                                "Answer only from the supplied context."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                )

            answer = response.choices[0].message.content

            st.subheader("🤖 Answer")
            st.write(answer)

            st.subheader("🔎 Retrieved sources")
            for number, result in enumerate(results, start=1):
                page = (
                    str(result["page"])
                    if result["page"] is not None
                    else "N/A"
                )

                with st.expander(
                    f"{number}. {result['filename']} — Page: {page}"
                ):
                    st.caption(
                        f"Hybrid score: {result['score']:.3f} | "
                        f"Semantic: {result['semantic_score']:.3f} | "
                        f"Keyword: {result['keyword_score']:.3f}"
                    )
                    st.write(result["text"])


st.divider()
st.caption(
    "Privacy note: document text is processed by this app. "
    "Only the retrieved chunks are sent to Groq for question answering."
)
'''

requirements = r'''streamlit
pypdf
python-docx
sentence-transformers
faiss-cpu
numpy
groq
gdown
'''

readme = r'''# 📄 AI Document Assistant

A simple Streamlit RAG-style document assistant that supports:

- PDF
- DOCX
- TXT
- Markdown (`.md`)
- Public Google Drive file/folder links
- Text extraction
- Overlapping text chunking
- Sentence Transformers embeddings
- FAISS semantic search
- Keyword search
- Hybrid search
- Groq question answering
- Retrieved source display
- Streamlit session-state caching so document embeddings are not recreated for every question

## 1. Project files

```text
ai-document-assistant/
├── app.py
├── requirements.txt
└── readme.md
