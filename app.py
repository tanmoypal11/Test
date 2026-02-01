import streamlit as st
import requests
import os

st.set_page_config(
    page_title="LLM Crop Diagnosis Test",
    page_icon="🌱",
    layout="centered"
)

st.title("🧠 LLM Crop Diagnosis (Test Only)")
st.write("Testing Hugging Face Inference Router with small model (CPU friendly)")

# ---------------- USER INPUT ----------------
default_prompt = """Apple___Apple_scab detected with 98.33% confidence

Tasks:
1. Explain the disease(s) in simple farmer-friendly language.
2. Mention possible causes.
3. Suggest immediate preventive or corrective actions.
4. If confidence is below 60%, mention uncertainty politely.

Keep the response concise and practical.
"""

user_prompt = st.text_area(
    "Input for LLM",
    value=default_prompt,
    height=220
)

# ---------------- HF CONFIG ----------------
HF_MODEL = "google/flan-t5-small"  # CPU-friendly
HF_API_URL = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL}"

HF_TOKEN = st.secrets.get("HF_TOKEN") or os.getenv("HF_TOKEN")
if not HF_TOKEN:
    st.error("❌ HF_TOKEN missing. Add it to Streamlit Secrets.")
    st.stop()

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

# ---------------- RUN LLM ----------------
if st.button("▶ Generate Explanation"):
    with st.spinner("Calling LLM..."):
        payload = {
            "inputs": f"""
You are an agriculture expert.

Rules:
- ONLY crop disease
- Simple farmer language
- Bullet points
- No unrelated topics

{user_prompt}
""",
            "parameters": {
                "max_new_tokens": 180,
                "temperature": 0.2,
                "top_p": 0.9
            }
        }

        response = requests.post(
            HF_API_URL,
            headers=HEADERS,
            json=payload,
            timeout=60
        )

    if response.status_code == 200:
        output = response.json()
        st.subheader("🧾 LLM Output")
        st.write(output[0]["generated_text"])
    else:
        st.error("❌ LLM call failed")
        st.code(response.text)
