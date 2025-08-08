import os, tempfile, requests, mimetypes, pathlib
from typing import List
import faiss
import numpy as np
from PyPDF2 import PdfReader
from groq import Groq

try:
    import docx  # python-docx
except ImportError:
    docx = None

# Load Groq API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("Please set GROQ_API_KEY in .env")

groq_client = Groq(api_key=GROQ_API_KEY)

# ----------------------------------------------------
# 1.  File helpers
# ----------------------------------------------------
def download_file(url: str) -> str:
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
# 2.  Chunking
# ----------------------------------------------------
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    start = 0
    chunks = []
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

# ----------------------------------------------------
# 3.  Remote Embeddings via Groq
# ----------------------------------------------------
def embed_texts(texts: List[str]) -> np.ndarray:
    """Use Groq's embedding API instead of local model."""
    embeddings = []
    for txt in texts:
        resp = groq_client.embeddings.create(
            model="llama3-8b-8192",  # Replace with Groq's embedding model name
            input=txt
        )
        embeddings.append(resp.data[0].embedding)
    return np.array(embeddings, dtype="float32")

def embed_single(text: str) -> List[float]:
    resp = groq_client.embeddings.create(
        model="llama3-8b-8192",  # Replace with Groq's embedding model name
        input=text
    )
    return resp.data[0].embedding

# ----------------------------------------------------
# 4.  FAISS
# ----------------------------------------------------
def build_faiss(embeddings: np.ndarray):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index
