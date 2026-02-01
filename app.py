import streamlit as st
import requests
import os

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="LLM Crop Diagnosis Test",
    page_icon="🌱",
    layout="centered"
)

st.title("🧠 LLM Crop Diagnosis (Test Only)")
st.write("This app tests ONLY the language model response.")

# --------------------------------------------------
# USER INPUT
# --------------------------------------------------
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

# --------------------------------------------------
# HUGGING FACE CONFIG
# --------------------------------------------------
HF_MODEL = "google/gemma-2b-it"   # you can change to mistralai/Mistral-7B-Instruct
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

HEADERS = {
    "Authorization": f"Bearer {st.secrets['HF_TOKEN']}",
    "Content-Type": "application/json"
}

# --------------------------------------------------
# RUN LLM
# --------------------------------------------------
if st.button("▶ Generate Explanation"):
    with st.spinner("Calling LLM..."):
        payload = {
            "inputs": f"""
You are an agriculture expert.

{user_prompt}

Rules:
- Only talk about crop disease
- Simple farmer language
- Bullet points
- No unrelated topics
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
        st.error("LLM call failed")
        st.code(response.text)
