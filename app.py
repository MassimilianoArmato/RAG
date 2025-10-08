import streamlit as st
import requests
import base64

# 🔧 Configurazione base
st.set_page_config(page_title="Screening CV con LLM", layout="centered")

st.title("📄 Screening CV intelligente")
st.markdown("Carica il tuo CV e confrontalo con un ruolo aziendale per ricevere un feedback personalizzato.")

# 📤 Upload del CV
uploaded_file = st.file_uploader("Carica il tuo CV (PDF o TXT)", type=["pdf", "txt"])

# 🎯 Selezione ruolo aziendale
roles = ["Machine Learning Engineer","Data Scientist", "Backend Developer", "IT Support", "HR Specialist", "Project Manager"]
selected_role = st.selectbox("Ruolo da confrontare", roles)

# 🧠 Invio al backend
if st.button("Analizza CV") and uploaded_file and selected_role:
    with st.spinner("Analisi in corso..."):
        # 🔁 Conversione file in base64
        file_bytes = uploaded_file.read()
        encoded_file = base64.b64encode(file_bytes).decode("utf-8")

        # 📡 Chiamata API
        response = requests.post(
            "http://localhost:8000/screening",
            json={
                "filename": uploaded_file.name,
                "filedata": encoded_file,
                "role": selected_role
            }
        )

        if response.status_code == 200:
            result = response.json()
            st.success("✅ Analisi completata!")
            st.subheader("📊 Feedback")
            st.markdown(result["feedback"])
        else:
            st.error("❌ Errore durante l'analisi. Controlla il backend.")

# 🧾 Footer
st.markdown("---")
st.caption("Powered by LangChain + FastAPI + LLM open-source")