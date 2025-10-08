import fitz  # PyMuPDF
import os

def extract_text_from_pdf(filepath: str) -> str:
    """Estrae testo da un file PDF usando PyMuPDF"""
    text = ""
    try:
        with fitz.open(filepath) as doc:
            for page in doc:
                text += page.get_text()
        return text.strip()
    except Exception as e:
        raise RuntimeError(f"Errore nel parsing PDF: {e}")

def extract_text_from_txt(filepath: str) -> str:
    """Estrae testo da un file TXT"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        raise RuntimeError(f"Errore nel parsing TXT: {e}")

def parse_cv(filepath: str) -> str:
    """Parsing generico del CV in base all'estensione"""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(filepath)
    elif ext == ".txt":
        return extract_text_from_txt(filepath)
    else:
        raise ValueError("Formato file non supportato. Usa PDF o TXT.")

def reduce_cv_text(cv_text: str, max_chars: int = 1200) -> str:
    """Estrae sezioni chiave dal CV per ridurre il prompt"""
    keywords = ["Esperienza", "Competenze", "Formazione", "Lingue", "Certificazioni"]
    lines = cv_text.splitlines()
    filtered = [line for line in lines if any(k.lower() in line.lower() for k in keywords)]
    reduced = "\n".join(filtered)
    return reduced[:max_chars]