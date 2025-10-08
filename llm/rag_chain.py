from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import logging
import torch

logging.basicConfig(level=logging.INFO)

MODEL_ID = "microsoft/phi-2"

try:
    logging.info(f"🔁 Caricamento modello: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    model.to("cpu")
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)
    logging.info("✅ Modello caricato correttamente su CPU")
except Exception as e:
    logging.error(f"❌ Errore nel caricamento del modello: {e}")
    generator = None

def build_prompt(cv_text: str, job_description: str, role: str = "") -> str:
    advanced_terms = ["LangChain", "LLM", "FastAPI", "modular", "agent", "orchestrazione", "deployment"]
    is_advanced = any(term.lower() in cv_text.lower() for term in advanced_terms)

    if "data scientist" in role.lower():
        intro = (
            "Sei un recruiter tecnico esperto in data science, machine learning e analisi statistica.\n"
            "Valuta il CV in base a competenze in Python, ML, deployment, orchestrazione e impatto scientifico."
        )
    elif "backend" in role.lower():
        intro = (
            "Sei un recruiter tecnico esperto in sviluppo backend, API REST, orchestrazione e architetture modulari.\n"
            "Valuta il CV in base a competenze in Python, FastAPI, LangChain, logging, error handling e scalabilità."
        )
    elif is_advanced:
        intro = (
            "Sei un recruiter specializzato in architetture LLM, agenti modulari e orchestrazione.\n"
            "Analizza il CV con attenzione a LangChain, deployment, modularità e compatibilità con ambienti enterprise."
        )
    else:
        intro = "Sei un recruiter professionista. Analizza il CV e confrontalo con la job description."

    prompt = (
        f"{intro}\nRispondi esclusivamente in italiano.\n"
        "Fornisci un feedback professionale in 3 sezioni:\n"
        "1. Compatibilità tecnica con il ruolo\n"
        "2. Competenze evidenziate (linguaggi, framework, architettura, progetti)\n"
        "3. Suggerimenti per migliorare il profilo\n\n"
        f"📄 CV:\n{cv_text}\n\n"
        f"🎯 Job Description:\n{job_description}\n\n"
        "✍️ Risposta:"
    )

    return prompt

def generate_feedback(cv_text: str, job_description: str, role: str = "") -> str:
    if generator is None:
        raise RuntimeError("Modello non disponibile. Verifica il caricamento.")

    try:
        prompt = build_prompt(cv_text, job_description, role)

        tokens = tokenizer(prompt, truncation=True, max_length=1200, return_tensors="pt")
        prompt = tokenizer.decode(tokens["input_ids"][0])

        with torch.no_grad():
            output = generator(
                prompt,
                max_new_tokens=128,
                do_sample=False,
                temperature=0.5,
                return_full_text=False
            )

        if not output or "generated_text" not in output[0]:
            raise ValueError("Output malformato dal modello.")
        return output[0]["generated_text"].strip()

    except Exception as e:
        logging.error(f"❌ Errore durante la generazione: {e}")
        raise RuntimeError(f"Errore nella generazione del feedback: {e}")