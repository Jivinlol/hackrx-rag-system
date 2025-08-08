from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
import os, asyncio, tempfile
from dotenv import load_dotenv
from groq import Groq
from utils import (
    download_file, extract_text, chunk_text,
    embed_texts, embed_single, build_faiss
)

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("Please set GROQ_API_KEY in .env")

groq_client = Groq(api_key=GROQ_API_KEY)
app = FastAPI(title="HackRx RAG System")

# ---------- API Schemas ----------
class HackRxRequest(BaseModel):
    documents: str  # URL to a single document
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]

# ---------- Endpoint -------------
@app.post("/hackrx/run", response_model=HackRxResponse)
async def run_submission(payload: HackRxRequest):
    try:
        # 1) Fetch & read the document -------------------
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            local_path = tmp_file.name

        await asyncio.to_thread(download_file, payload.documents, local_path)
        full_text = await asyncio.to_thread(extract_text, local_path)

        # Cleanup temp file after use
        try:
            os.remove(local_path)
        except FileNotFoundError:
            pass

        # 2) Split & embed -------------------------------
        chunks = chunk_text(full_text, chunk_size=500, overlap=75)
        chunk_emb = await asyncio.to_thread(embed_texts, chunks)
        index = build_faiss(chunk_emb)

        # 3) Function to answer a single question --------
        async def answer_question(q: str) -> str:
            try:
                q_emb = embed_single(q).reshape(1, -1)
                _, idx = index.search(q_emb, k=3)
                context = "\n---\n".join([chunks[i] for i in idx[0]])

                prompt = (
                    "You are an insurance policy assistant. "
                    "Answer ONLY from the context below.\n\n"
                    f"Context:\n{context}\n\n"
                    f"Question: {q}\nAnswer:"
                )

                resp = await asyncio.to_thread(
                    groq_client.chat.completions.create,
                    model="llama3-8b-8192",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=400
                )
                return resp.choices[0].message.content.strip()

            except Exception as e:
                return f"Error answering question: {str(e)}"

        # 4) Answer all questions in parallel ------------
        answers = await asyncio.gather(*[answer_question(q) for q in payload.questions])

        return {"answers": answers}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
