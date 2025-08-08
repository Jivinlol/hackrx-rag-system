import os, tempfile, requests, mimetypes, pathlib
from typing import List, Tuple
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
try:
    import docx  # python-docx
except ImportError:
    docx = None  # DOCX extraction will be skipped if not available

# --- Local embedding model (≈90 MB) -----------------
print("Loading sentence-transformer model (all-MiniLM-L6-v2)…")
_ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
print("✅  Embedding model loaded.")

# ----------------------------------------------------
# 1.  File helpers
# ----------------------------------------------------
def download_file(url: str) -> str:
    """Download to a temp file and return local path."""
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    suffix = pathlib.Path(url.split("?")[0]).suffix or ".tmp"
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(resp.content)
    return path

def load_pdf(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join([p.extract_text() for p in reader.pages if p.extract_text()])

def load_docx(path: str) -> str:
    if docx is None:
        raise RuntimeError("python-docx not installed")
    d = docx.Document(path)
    return "\n".join([p.text for p in d.paragraphs])

def extract_text(path: str) -> str:
    mtype = mimetypes.guess_type(path)[0] or ""
    if path.lower().endswith(".pdf") or "pdf" in mtype:
        return load_pdf(path)
    if path.lower().endswith(".docx") or "word" in mtype:
        return load_docx(path)
    raise ValueError(f"Unsupported file type for {path}")

# ----------------------------------------------------
# 2.  Chunking & Embeddings
# ----------------------------------------------------
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    start = 0
    chunks = []
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def embed_texts(texts: List[str]) -> np.ndarray:
    """Return float32 numpy array (n_chunks × dim)."""
    emb = _ST_MODEL.encode(texts, show_progress_bar=False)
    return np.array(emb, dtype="float32")

def embed_single(text: str) -> List[float]:
    return _ST_MODEL.encode([text])[0].tolist()

# ----------------------------------------------------
# 3.  FAISS
# ----------------------------------------------------
def build_faiss(embeddings: np.ndarray):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index
