from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
import os
import logging
import time

from services.parser import parse_cv
from retrieval.retriever import retrieve_similar_role, load_job_descriptions
from llm.rag_chain import generate_feedback

logging.basicConfig(level=logging.INFO)

app = FastAPI()
UPLOAD_FOLDER = "uploaded_cv"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class CVRequest(BaseModel):
    filename: str
    filedata: str  # base64
    role: str

@app.post("/screening")
def screen_cv(request: CVRequest):
    try:
        total_start = time.time()
        logging.info(f"📥 Ricezione file: {request.filename}")
        file_bytes = base64.b64decode(request.filedata)
        filepath = os.path.join(UPLOAD_FOLDER, request.filename)
        with open(filepath, "wb") as f:
            f.write(file_bytes)
        logging.info("✅ File salvato")

        # ⏱️ Parsing CV
        parse_start = time.time()
        cv_text = parse_cv(filepath)
        parse_time = time.time() - parse_start
        logging.info(f"📄 CV parsed: {len(cv_text)} caratteri in {parse_time:.2f} sec")

        # ⏱️ Retrieval ruolo
        retrieval_start = time.time()
        best_role, similarity = retrieve_similar_role(cv_text)
        retrieval_time = time.time() - retrieval_start
        logging.info(f"🔍 Ruolo più simile: {best_role} (similarità: {similarity:.2f}) in {retrieval_time:.2f} sec")

        job_descriptions = load_job_descriptions()
        job_description = job_descriptions.get(best_role, "")
        if not job_description:
            raise ValueError(f"Job description mancante per il ruolo: {best_role}")

        # ⏱️ Generazione feedback
        gen_start = time.time()
        feedback_text = generate_feedback(cv_text, job_description, role=best_role)
        gen_time = time.time() - gen_start
        logging.info(f"🧠 Generazione completata in {gen_time:.2f} sec")

        total_time = time.time() - total_start
        logging.info(f"⏱️ Tempo totale: {total_time:.2f} sec")

        return {
            "feedback": feedback_text,
            "role_matched": best_role,
            "similarity": float(similarity),
            "timing": {
                "parse_time": round(parse_time, 2),
                "retrieval_time": round(retrieval_time, 2),
                "generation_time": round(gen_time, 2),
                "total_time": round(total_time, 2)
            }
        }

    except Exception as e:
        logging.error(f"❌ Errore nel backend: {e}")
        raise HTTPException(status_code=500, detail=f"Errore durante l’analisi: {e}")