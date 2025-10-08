import json
import os
import logging
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL = SentenceTransformer(MODEL_ID, device="cpu")

INDEX_PATH = "data/faiss_index.bin"
ROLES_PATH = "data/roles.json"

def reduce_cv_text(cv_text: str, max_chars: int = 1200) -> str:
    keywords = ["Esperienza", "Competenze", "Obiettivo", "GitHub", "LangChain", "Machine Learning", "FastAPI"]
    lines = cv_text.splitlines()
    filtered = [line for line in lines if any(k.lower() in line.lower() for k in keywords)]
    reduced = "\n".join(filtered)
    return reduced[:max_chars]

def load_job_descriptions():
    with open("data/job_descriptions.json", "r", encoding="utf-8") as f:
        return json.load(f)

def retrieve_similar_role(cv_text: str):
    if not os.path.exists(INDEX_PATH) or not os.path.exists(ROLES_PATH):
        raise FileNotFoundError("Indice FAISS o file dei ruoli mancante.")

    reduced_text = reduce_cv_text(cv_text)
    query_embedding = EMBEDDING_MODEL.encode([reduced_text], batch_size=8)
    query_embedding = np.array(query_embedding).astype("float32")

    index = faiss.read_index(INDEX_PATH)
    with open(ROLES_PATH, "r", encoding="utf-8") as f:
        roles = json.load(f)

    D, I = index.search(query_embedding, k=1)
    best_index = I[0][0]
    similarity = float(D[0][0])
    best_role = roles[str(best_index)]

    logging.info(f"🔍 Retrieval: ruolo={best_role}, similarità={similarity:.2f}")

    if similarity < 0.65:
        logging.warning("⚠️ Similarità bassa, fallback su ruolo selezionato manualmente.")
        return "Machine Learning Engineer", 0.0

    return best_role, similarity

def build_faiss_index():
    logging.info("🔧 Avvio costruzione indice FAISS")
    job_descriptions = load_job_descriptions()

    roles = list(job_descriptions.keys())
    descriptions = list(job_descriptions.values())

    embeddings = EMBEDDING_MODEL.encode(descriptions, batch_size=8, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    logging.info(f"✅ Indice FAISS salvato in {INDEX_PATH}")

    with open(ROLES_PATH, "w", encoding="utf-8") as f:
        json.dump({str(i): role for i, role in enumerate(roles)}, f, indent=2, ensure_ascii=False)
    logging.info(f"✅ Ruoli salvati in {ROLES_PATH}")