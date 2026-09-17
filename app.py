import io
import os
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import faiss
import gdown
import numpy as np
import streamlit as st
from docx import Document
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# =========================================================
# Page setup
# =========================================================
st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# Custom styling
# =========================================================
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .app-header {
        padding: 1.6rem 2rem;
        border-radius: 16px;
        background: linear-gradient(120deg, #4f46e5 0%, #7c3aed 50%, #a855f7 100%);
        color: white;
        margin-bottom: 1.4rem;
        box-shadow: 0 8px 24px rgba(124, 58, 237, 0.25);
    }
    .app-header h1 { margin: 0; font-size: 1.9rem; }
    .app-header p { margin: 0.3rem 0 0 0; opacity: 0.9; font-size: 0.95rem; }

    .metric-card {
        background: white;
        border: 1px solid #eef0f4;
        border-radius: 14px;
        padding: 1rem 1.1rem;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
    }
    .metric-card .label { font-size: 0.78rem; color: #6b7280; text-transform: uppercase; letter-spacing: .04em; }
    .metric-card .value { font-size: 1.6rem; font-weight: 700; color: #1f2937; }

    .doc-row {
        display: flex; align-items: center; gap: 0.7rem;
        padding: 0.6rem 0.8rem; border-radius: 10px;
        background: #f9fafb; margin-bottom: 0.45rem;
        border: 1px solid #f0f1f4;
    }
    .doc-badge {
        font-size: 0.68rem; font-weight: 700; padding: 0.15rem 0.5rem;
        border-radius: 6px; color: white; letter-spacing: .03em;
    }
    .badge-pdf { background: #ef4444; }
    .badge-docx { background: #2563eb; }
    .badge-txt { background: #6b7280; }
    .badge-md { background: #16a34a; }

    .status-ok { color: #16a34a; font-weight: 600; }
    .status-warn { color: #d97706; font-weight: 600; }
    .status-err { color: #dc2626; font-weight: 600; }

    .score-bar-bg { background: #eef0f4; border-radius: 6px; height: 7px; width: 100%; }
    .score-bar-fill { background: linear-gradient(90deg, #7c3aed, #a855f7); height: 7px; border-radius: 6px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-header">
        <h1>📄 AI Document Assistant</h1>
        <p>PDF • DOCX • TXT • MD &nbsp;|&nbsp; Hybrid semantic + keyword search &nbsp;|&nbsp; Powered by Groq</p>
    </div>
    """,
    unsafe_allow_html=True,
)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
BADGE_CLASS = {".pdf": "badge-pdf", ".docx": "badge-docx", ".txt": "badge-txt", ".md": "badge-md"}

GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3-32b",
]

EMBEDDING_MODELS = {
    "MiniLM-L6 (fast, recommended)": "all-MiniLM-L6-v2",
    "MPNet-base (higher accuracy, slower)": "all-mpnet-base-v2",
}


# =========================================================
# Cached resources
# =========================================================
@st.cache_resource(show_spinner=False)
def load_embedding_model(model_name):
    return SentenceTransformer(model_name)


# =========================================================
# Document extraction (each wrapped so one bad file can't kill the batch)
# =========================================================
def extract_pdf(file_bytes, filename):
    reader = PdfReader(io.BytesIO(file_bytes))

    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception:
            raise ValueError("PDF is password-protected and could not be opened.")

    records = []
    failed_pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            failed_pages.append(page_number)
            continue

        if text.strip():
            records.append({"text": text.strip(), "filename": filename, "page": page_number})

    if not records:
        note = " (this looks like a scanned/image-only PDF with no extractable text; OCR is not supported)"
        raise ValueError(f"No extractable text found in {len(reader.pages)} page(s){note}.")

    return records, failed_pages


def extract_docx(file_bytes, filename):
    document = Document(io.BytesIO(file_bytes))

    parts = [p.text.strip() for p in document.paragraphs if p.text.strip()]

    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                parts.append(" | ".join(cells))

    text = "\n".join(parts)
    if not text:
        raise ValueError("No readable text found in this DOCX file.")

    return [{"text": text, "filename": filename, "page": None}], []


def extract_txt(file_bytes, filename):
    text = file_bytes.decode("utf-8", errors="ignore").strip()
    if not text:
        raise ValueError("File is empty.")
    return [{"text": text, "filename": filename, "page": None}], []


def extract_md(file_bytes, filename):
    text = file_bytes.decode("utf-8", errors="ignore").strip()
    if not text:
        raise ValueError("File is empty.")
    return [{"text": text, "filename": filename, "page": None}], []


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

    raise ValueError(f"Unsupported file type: {extension or 'unknown'}")


# =========================================================
# Chunking
# =========================================================
def chunk_text(records, chunk_size=800, overlap=120):
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
                chunks.append({"text": chunk, "filename": record["filename"], "page": record["page"]})

            if end >= len(text):
                break
            start = end - overlap

    return chunks


# =========================================================
# Embeddings + FAISS
# (index is built once and kept as a live object in session_state —
#  no lossy byte round-trip, which is what caused prior crashes)
# =========================================================
def create_embeddings(chunks, model):
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False).astype("float32")
    faiss.normalize_L2(embeddings)
    return embeddings


def build_document_store(chunks, model):
    embeddings = create_embeddings(chunks, model)
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return {"chunks": chunks, "index": index, "dimension": dimension}


# =========================================================
# Keyword search
# =========================================================
STOP_WORDS = {
    "the", "and", "for", "with", "from", "this", "that", "what", "when",
    "where", "which", "who", "how", "why", "are", "was", "were", "is",
    "to", "of", "in", "on", "a", "an", "as", "by", "or", "it", "be",
    "about", "can", "does", "do", "i", "me", "my", "please", "tell",
}


def important_words(question):
    words = re.findall(r"\b[a-zA-Z0-9][a-zA-Z0-9_-]*\b", question.lower())
    return [w for w in words if w not in STOP_WORDS and len(w) > 1]


def keyword_score(question, text):
    words = important_words(question)
    if not words:
        return 0.0
    text_lower = text.lower()
    matches = sum(1 for w in words if w in text_lower)
    return matches / len(words)


def hybrid_search(question, store, model, top_k=5):
    question_embedding = model.encode([question], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(question_embedding)

    candidate_k = min(max(top_k * 4, 10), len(store["chunks"]))
    semantic_scores, indices = store["index"].search(question_embedding, candidate_k)

    candidates = []
    for semantic_score, idx in zip(semantic_scores[0], indices[0]):
        if idx < 0:
            continue

        chunk = store["chunks"][idx]
        kw_score = keyword_score(question, chunk["text"])
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


# =========================================================
# Google Drive loader (fixed)
#
# Bug in the previous version: single-file links were downloaded to a
# filename with NO extension, so the extension filter always rejected
# them and the app reported "no supported files" or silently failed.
#
# Fix: let gdown resolve the real filename (and extension) itself by
# pointing it at a directory instead of a fixed filename, and give
# clear, specific errors instead of a generic failure.
# =========================================================
def parse_drive_url(url):
    folder_match = re.search(r"/folders/([a-zA-Z0-9_-]+)", url)
    if folder_match:
        return "folder", folder_match.group(1)

    file_match = re.search(r"/file/d/([a-zA-Z0-9_-]+)", url) or re.search(r"[?&]id=([a-zA-Z0-9_-]+)", url)
    if file_match:
        return "file", file_match.group(1)

    return None, None


def load_from_google_drive(url):
    kind, drive_id = parse_drive_url(url)

    if kind is None:
        raise ValueError(
            "Couldn't recognize that as a Google Drive link. Use a file link "
            "(.../file/d/FILE_ID/view) or a folder link (.../drive/folders/FOLDER_ID)."
        )

    temp_dir = tempfile.mkdtemp(prefix="drive_docs_")

    try:
        if kind == "folder":
            try:
                downloaded_paths = gdown.download_folder(
                    id=drive_id, output=temp_dir, quiet=True, use_cookies=False
                )
            except Exception as exc:
                raise RuntimeError(
                    "Couldn't open that Drive folder. Make sure sharing is set to "
                    f"'Anyone with the link'. ({exc})"
                ) from exc

            if not downloaded_paths:
                raise RuntimeError(
                    "The folder appears to be empty, or it isn't shared publicly "
                    "('Anyone with the link' must be enabled)."
                )
        else:
            try:
                # Trailing separator tells gdown to treat this as a directory and
                # infer the real filename (with correct extension) itself.
                out = gdown.download(id=drive_id, output=temp_dir + os.sep, quiet=True, fuzzy=True)
            except Exception as exc:
                raise RuntimeError(f"Couldn't download the Drive file. ({exc})") from exc

            if not out:
                raise RuntimeError(
                    "Couldn't download that file. Make sure sharing is set to "
                    "'Anyone with the link'."
                )
            downloaded_paths = [out]

        loaded, skipped = [], []
        for item in downloaded_paths:
            path = Path(item)
            if not path.is_file():
                continue
            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                loaded.append((path.name, path.read_bytes()))
            else:
                skipped.append(path.name)

        return loaded, skipped

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# =========================================================
# Process documents
# =========================================================
def process_documents(document_items, model, chunk_size, overlap, progress_cb=None):
    all_records = []
    document_info = []

    total = len(document_items)
    for i, (filename, file_bytes) in enumerate(document_items, start=1):
        extension = Path(filename).suffix.lower()
        entry = {
            "filename": filename,
            "type": extension.replace(".", "").upper() or "?",
            "status": "ok",
            "message": "",
            "characters": 0,
            "pages": None,
        }

        try:
            records, failed_pages = extract_document(file_bytes, filename)
            all_records.extend(records)
            entry["characters"] = sum(len(r["text"]) for r in records)

            if extension == ".pdf":
                entry["pages"] = len(records)

            if failed_pages:
                entry["status"] = "warn"
                entry["message"] = f"{len(failed_pages)} page(s) couldn't be parsed."

        except Exception as exc:
            entry["status"] = "err"
            entry["message"] = str(exc)

        document_info.append(entry)

        if progress_cb:
            progress_cb(i / total, filename)

    chunks = chunk_text(all_records, chunk_size=chunk_size, overlap=overlap)

    if not chunks:
        return None, document_info

    store = build_document_store(chunks, model)
    return store, document_info


# =========================================================
# Session state
# =========================================================
defaults = {
    "document_store": None,
    "document_info": [],
    "chat_history": [],
    "processing_log": [],
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# Sidebar
# =========================================================
with st.sidebar:
    st.subheader("📚 Add documents")

    uploaded_files = st.file_uploader(
        "Upload files",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
    )

    st.divider()
    st.subheader("🔗 Google Drive")
    drive_url = st.text_input(
        "Public Drive file or folder link",
        placeholder="https://drive.google.com/...",
    )
    st.caption("Link sharing must be set to **'Anyone with the link'**. Supported: PDF, DOCX, TXT, MD.")

    st.divider()
    with st.expander("⚙️ Advanced settings"):
        embedding_choice = st.selectbox("Embedding model", list(EMBEDDING_MODELS.keys()))
        groq_model = st.selectbox("Groq model", GROQ_MODELS, index=0)
        custom_model = st.text_input("...or type a custom Groq model id", value="")
        temperature = st.slider("Answer creativity (temperature)", 0.0, 1.0, 0.0, 0.1)
        chunk_size = st.slider("Chunk size (characters)", 400, 1500, 800, 50)
        overlap = st.slider("Chunk overlap", 50, 300, 120, 10)
        top_k = st.slider("Sources per answer", 1, 10, 5)

    process_button = st.button("⚙️ Process documents", use_container_width=True, type="primary")
    reset_button = st.button("🗑️ Clear everything", use_container_width=True)

    if reset_button:
        for key, value in defaults.items():
            st.session_state[key] = value
        st.rerun()

active_groq_model = custom_model.strip() if custom_model.strip() else groq_model


# =========================================================
# Process button logic
# =========================================================
if process_button:
    document_items = []

    if uploaded_files:
        document_items.extend([(f.name, f.getvalue()) for f in uploaded_files])

    drive_skipped = []
    if drive_url.strip():
        with st.spinner("Connecting to Google Drive..."):
            try:
                drive_items, drive_skipped = load_from_google_drive(drive_url.strip())
                document_items.extend(drive_items)
                if not drive_items and not drive_skipped:
                    st.warning("No files were found at that Drive link.")
            except Exception as exc:
                st.error(f"❌ Google Drive: {exc}")

    if drive_skipped:
        st.info(f"Skipped unsupported Drive file(s): {', '.join(drive_skipped)}")

    if not document_items:
        st.warning("Upload at least one document or provide a working Drive link.")
    else:
        model = load_embedding_model(EMBEDDING_MODELS[embedding_choice])

        progress_bar = st.progress(0.0, text="Starting...")

        def _update_progress(fraction, filename):
            progress_bar.progress(fraction, text=f"Reading {filename}...")

        store, info = process_documents(
            document_items, model, chunk_size, overlap, progress_cb=_update_progress
        )
        progress_bar.empty()

        st.session_state.document_info = info
        st.session_state.processing_log = info

        if store is None:
            st.error("No text could be extracted from any of the supplied documents.")
        else:
            st.session_state.document_store = store
            ok_count = sum(1 for i in info if i["status"] == "ok")
            st.success(f"✅ Processed {ok_count}/{len(info)} document(s) into {len(store['chunks'])} chunks.")


# =========================================================
# Main layout: tabs
# =========================================================
tab_docs, tab_chat = st.tabs(["📁 Documents", "💬 Chat"])

# ---- Documents tab ----
with tab_docs:
    store = st.session_state.document_store
    info = st.session_state.document_info

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f'<div class="metric-card"><div class="label">Documents</div>'
            f'<div class="value">{len(info)}</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        chunks_n = len(store["chunks"]) if store else 0
        st.markdown(
            f'<div class="metric-card"><div class="label">Chunks</div>'
            f'<div class="value">{chunks_n}</div></div>',
            unsafe_allow_html=True,
        )
    with col3:
        chars_n = sum(i["characters"] for i in info)
        st.markdown(
            f'<div class="metric-card"><div class="label">Characters</div>'
            f'<div class="value">{chars_n:,}</div></div>',
            unsafe_allow_html=True,
        )
    with col4:
        ok_n = sum(1 for i in info if i["status"] == "ok")
        st.markdown(
            f'<div class="metric-card"><div class="label">Ready</div>'
            f'<div class="value">{ok_n}/{len(info)}</div></div>',
            unsafe_allow_html=True,
        )

    st.write("")

    if not info:
        st.info("Upload documents or add a Google Drive link in the sidebar, then click **Process documents**.")
    else:
        st.subheader("Extraction results")
        for item in info:
            badge_class = BADGE_CLASS.get(f".{item['type'].lower()}", "badge-txt")

            if item["status"] == "ok":
                status_html = '<span class="status-ok">✓ processed</span>'
            elif item["status"] == "warn":
                status_html = f'<span class="status-warn">⚠ {item["message"]}</span>'
            else:
                status_html = f'<span class="status-err">✗ {item["message"]}</span>'

            page_info = f" · {item['pages']} pages" if item.get("pages") else ""

            st.markdown(
                f'<div class="doc-row">'
                f'<span class="doc-badge {badge_class}">{item["type"]}</span>'
                f'<div style="flex:1;">'
                f'<b>{item["filename"]}</b>'
                f'<div style="font-size:0.8rem;color:#6b7280;">{item["characters"]:,} characters{page_info}</div>'
                f'</div>'
                f'<div>{status_html}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

# ---- Chat tab ----
with tab_chat:
    store = st.session_state.document_store

    if not store:
        st.info("Process some documents first, then come back here to ask questions.")
    else:
        header_col, clear_col, export_col = st.columns([6, 1, 1.4])
        with header_col:
            st.caption(f"Answering from **{len(store['chunks'])} chunks** using `{active_groq_model}`")
        with clear_col:
            if st.button("Clear chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        with export_col:
            if st.session_state.chat_history:
                transcript = "\n\n".join(
                    f"{'Q' if turn['role']=='user' else 'A'}: {turn['content']}"
                    for turn in st.session_state.chat_history
                )
                st.download_button(
                    "Export chat",
                    data=transcript,
                    file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                    use_container_width=True,
                )

        for turn in st.session_state.chat_history:
            with st.chat_message(turn["role"]):
                st.write(turn["content"])
                if turn["role"] == "assistant" and turn.get("sources"):
                    with st.expander(f"🔎 {len(turn['sources'])} source(s)"):
                        for n, result in enumerate(turn["sources"], start=1):
                            page = str(result["page"]) if result["page"] is not None else "N/A"
                            pct = max(0, min(100, int(result["score"] * 100)))
                            st.markdown(f"**{n}. {result['filename']}** — page: {page}")
                            st.markdown(
                                f'<div class="score-bar-bg"><div class="score-bar-fill" '
                                f'style="width:{pct}%;"></div></div>',
                                unsafe_allow_html=True,
                            )
                            st.caption(
                                f"relevance {pct}% · semantic {result['semantic_score']:.2f} "
                                f"· keyword {result['keyword_score']:.2f}"
                            )
                            st.write(result["text"])
                            st.markdown("---")

        question = st.chat_input("Ask something about your documents...")

        if question:
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.write(question)

            model = load_embedding_model(EMBEDDING_MODELS[embedding_choice])

            with st.spinner("Searching documents..."):
                results = hybrid_search(question, store, model, top_k=top_k)

            context_parts = []
            for n, result in enumerate(results, start=1):
                page = f"page {result['page']}" if result["page"] is not None else "page unavailable"
                context_parts.append(f"[Source {n}: {result['filename']}, {page}]\n{result['text']}")
            context = "\n\n".join(context_parts)

            api_key = st.secrets.get("GROQ_API_KEY") if hasattr(st, "secrets") else None
            api_key = api_key or os.getenv("GROQ_API_KEY")

            with st.chat_message("assistant"):
                if not api_key:
                    error_msg = "GROQ_API_KEY is missing. Add it to Streamlit Secrets as `GROQ_API_KEY`."
                    st.error(error_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg, "sources": []})
                else:
                    client = Groq(api_key=api_key)

                    # include brief recent history so natural follow-ups work
                    recent_turns = st.session_state.chat_history[-6:-1]
                    history_snippet = "\n".join(
                        f"{'User' if t['role']=='user' else 'Assistant'}: {t['content']}" for t in recent_turns
                    )

                    prompt = f"""Answer the user's question using ONLY the context below.

Rules:
- Do not use outside knowledge.
- If the answer is not contained in the context, say: "The information is not available in the provided documents."
- Do not invent facts, citations, page numbers, or sources.
- Keep the answer clear and concise.

RECENT CONVERSATION (for follow-up context only, not a source of facts):
{history_snippet}

CONTEXT:
{context}

USER QUESTION:
{question}
"""

                    try:
                        with st.spinner("Generating answer..."):
                            response = client.chat.completions.create(
                                model=active_groq_model,
                                messages=[
                                    {
                                        "role": "system",
                                        "content": "You are a document question-answering assistant. Answer only from the supplied context.",
                                    },
                                    {"role": "user", "content": prompt},
                                ],
                                temperature=temperature,
                            )
                        answer = response.choices[0].message.content
                    except Exception as exc:
                        answer = f"⚠️ Groq request failed: {exc}"

                    st.write(answer)
                    if results:
                        with st.expander(f"🔎 {len(results)} source(s)"):
                            for n, result in enumerate(results, start=1):
                                page = str(result["page"]) if result["page"] is not None else "N/A"
                                pct = max(0, min(100, int(result["score"] * 100)))
                                st.markdown(f"**{n}. {result['filename']}** — page: {page}")
                                st.markdown(
                                    f'<div class="score-bar-bg"><div class="score-bar-fill" '
                                    f'style="width:{pct}%;"></div></div>',
                                    unsafe_allow_html=True,
                                )
                                st.caption(
                                    f"relevance {pct}% · semantic {result['semantic_score']:.2f} "
                                    f"· keyword {result['keyword_score']:.2f}"
                                )
                                st.write(result["text"])
                                st.markdown("---")

                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": answer, "sources": results}
                    )

st.divider()
st.caption(
    "Privacy note: document text is processed locally by this app. "
    "Only the retrieved chunks (not full documents) are sent to Groq to generate each answer."
)
