# 📄 AI Document Assistant (Advanced)

A Streamlit app that answers questions about your PDF, DOCX, TXT, and MD files using hybrid (semantic + keyword) search and Groq for generation. Upload files directly or pull them in from a public Google Drive link.

## What's new in this version

**Bugs fixed:**
- **FAISS crash on every question.** The old code serialized embeddings to raw bytes and tried to reshape them with `.shape[1]` on a 1‑D array — this always threw `IndexError` when you asked a question. Fixed by keeping the FAISS index as a live object in `st.session_state` instead of round‑tripping through bytes.
- **Google Drive single-file links failing / "no supported files" error.** The old code downloaded single files to a filename with no extension, so the `.pdf/.docx/.txt/.md` filter always rejected them. Fixed by letting `gdown` resolve the real filename (with the correct extension) itself, plus clearer error messages (private link, empty folder, bad URL, etc.) and support for more Drive link formats (`/file/d/…`, `?id=…`, `/folders/…`).
- **One bad file killing the whole batch.** Extraction is now wrapped per-file, so a corrupt or password-protected PDF just gets flagged with a status instead of crashing the app.
- **Scanned/image-only PDFs failing silently.** These are now detected and reported clearly (OCR isn't supported, so text-based PDFs are required).

**New / upgraded UI:**
- Polished, card-based layout with a gradient header and live metrics (documents, chunks, characters, success rate).
- Two tabs: **Documents** (upload status per file, with per-file error/warning messages) and **Chat** (a real chat interface using `st.chat_message` / `st.chat_input`).
- Chat history persists during the session, with relevance-score bars per retrieved source and a **transcript export** button.
- Sidebar **Advanced settings**: choice of embedding model (fast vs. more accurate), choice of Groq model (or type a custom model id), temperature, chunk size/overlap, and number of sources.
- One-click **Clear everything** to reset the session.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Add your Groq API key to Streamlit Secrets (`.streamlit/secrets.toml`):
   ```toml
   GROQ_API_KEY = "gsk_..."
   ```
   (An environment variable `GROQ_API_KEY` also works as a fallback.)
3. Run the app:
   ```bash
   streamlit run app.py
   ```

## Using Google Drive

- **Single file:** share it so "Anyone with the link" can view, then paste the link (`.../file/d/FILE_ID/view` or `?id=FILE_ID`).
- **Folder:** share the folder the same way, then paste the folder link (`.../drive/folders/FOLDER_ID`). All supported files inside are downloaded.
- Private files/folders (restricted to specific accounts) can't be accessed this way — for those you'd need a Drive API integration with OAuth/service-account credentials, which is out of scope for this simple app.

## How it works

1. **Extract** text from each uploaded/downloaded file (per-page for PDFs, including tables for DOCX).
2. **Chunk** the text into overlapping segments, keeping filename/page metadata.
3. **Embed** each chunk with a Sentence Transformers model and index it in FAISS (built once per "Process documents" click — not recreated per question).
4. **Search**: each question is embedded and matched against the index (semantic score), combined with a simple keyword-overlap score (70% semantic / 30% keyword).
5. **Answer**: the top matching chunks are sent to Groq as context, with instructions to answer only from that context.

## Notes

- Larger Drive folders / many large PDFs will take longer on the "Process documents" step — this is expected (embeddings are computed once, not per question).
- Only the retrieved chunks (not entire documents) are sent to Groq for each answer.
