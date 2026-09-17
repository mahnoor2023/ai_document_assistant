# 📄 AI Document Assistant

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
```

## 2. Install

Use Python 3.10 or newer.

```bash
pip install -r requirements.txt
```

## 3. Add the Groq API key

Do **not** put the API key inside `app.py`.

For local Streamlit, create:

```text
.streamlit/secrets.toml
```

Put this inside:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

You can also configure the same secret in Streamlit Community Cloud under the app's Secrets settings.

The app reads:

```python
st.secrets["GROQ_API_KEY"]
```

and never hardcodes the key.

## 4. Run

```bash
streamlit run app.py
```

## 5. How the pipeline works

```text
PDF / DOCX / TXT / MD / Google Drive
                ↓
          Text extraction
                ↓
       Overlapping chunks
                ↓
 Sentence Transformers embeddings
                ↓
          FAISS index
                ↓
       User's question
          ↙          ↘
   Semantic search   Keyword search
          ↘          ↙
          Hybrid ranking
                ↓
        Top relevant chunks
                ↓
       Groq LLM + context
                ↓
             Answer
                ↓
        Retrieved sources
```

## 6. Important design choice: embeddings are created once

When you click **Process documents**, the app:

1. Extracts the text.
2. Creates overlapping chunks.
3. Creates embeddings for all chunks.
4. Stores the embeddings and chunk metadata in `st.session_state`.

When you ask another question, the app does **not** recreate document embeddings.

It only:

1. Embeds the new question.
2. Searches the stored document vectors.
3. Runs keyword matching.
4. Combines both scores.
5. Sends the retrieved chunks to Groq.

The FAISS index is rebuilt from the already-stored embedding array when needed; the expensive document embedding step is not repeated.

## 7. Hybrid search

The app combines:

```text
Hybrid score = 0.70 × semantic score + 0.30 × keyword score
```

Semantic search finds chunks with similar meaning.

Keyword search checks whether important words from the question occur in the chunk.

The important-word extraction is intentionally simple and uses a small stop-word list, so the project remains easy to explain.

## 8. Metadata

Every chunk keeps:

```python
{
    "text": "...",
    "filename": "...",
    "page": 1
}
```

PDF chunks preserve the PDF page number.

DOCX, TXT and MD files do not have a reliable page concept at extraction time, so their page value is `None`.

## 9. Google Drive

Paste a **publicly accessible** Google Drive file or folder link into the sidebar.

The app uses `gdown` to download supported files and sends them through the same pipeline:

```text
Google Drive
→ extraction
→ chunking
→ embeddings
→ FAISS
→ hybrid search
→ Groq
```

Local file upload continues to work at the same time.

For private Google Drive content, a proper Google Drive API/OAuth integration is required. This simple version intentionally avoids adding OAuth complexity.

## 10. Groq answer behavior

The model is instructed to answer only from retrieved context.

If the context does not contain the answer, it should say:

> The information is not available in the provided documents.

The app also displays the retrieved chunks underneath every answer so you can inspect the evidence used by the RAG pipeline.

## 11. First run

The first use of Sentence Transformers may download the embedding model:

```text
all-MiniLM-L6-v2
```

After it is available locally, later runs can reuse the cached model.

## 12. Streamlit deployment

For Streamlit Community Cloud:

1. Push `app.py`, `requirements.txt`, and `readme.md` to GitHub.
2. Create a Streamlit app from the repository.
3. Open the app's **Secrets** settings.
4. Add:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

5. Deploy.

Do not commit `.streamlit/secrets.toml` to GitHub.

A useful `.gitignore` entry is:

```text
.streamlit/secrets.toml
__pycache__/
*.pyc
```
