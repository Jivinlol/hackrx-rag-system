from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np, os
from dotenv import load_dotenv
from groq import Groq

from utils import (
    download_file, extract_text, chunk_text,
    embed_texts, embed_single, build_faiss
)

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("Please set GROQ_API_KEY in .env")

groq_client = Groq(api_key=GROQ_API_KEY)

app = FastAPI(title="HackRx RAG System")

# ---------- API Schemas ----------
class HackRxRequest(BaseModel):
    documents: str                    # single URL for this challenge
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]

# ---------- Endpoint -------------
@app.post("/hackrx/run", response_model=HackRxResponse)
def run_submission(payload: HackRxRequest):
    try:
        # 1) Fetch & read the document -------------------
        local_path = download_file(payload.documents)
        full_text  = extract_text(local_path)

        # 2) Split & embed -------------------------------
        chunks = chunk_text(full_text, chunk_size=600, overlap=75)
        chunk_emb = embed_texts(chunks)
        index = build_faiss(chunk_emb)

        answers = []

        # 3) Loop over questions -------------------------
        for q in payload.questions:
            q_emb = np.array([embed_single(q)], dtype="float32")
            _, idx = index.search(q_emb, k=3)
            context = "\n---\n".join([chunks[i] for i in idx[0]])

            prompt = (
                "You are an insurance policy assistant. "
                "Answer ONLY from the context below.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {q}\nAnswer:"
            )

            resp = groq_client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=400
            )
            answer = resp.choices[0].message.content.strip()
            answers.append(answer)

        return {"answers": answers}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
