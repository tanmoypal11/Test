import streamlit as st
from groq import Groq

# 1. Page Config must be first
st.set_page_config(page_title="Plant Disease AI", page_icon="🌿")

# 2. Sidebar Settings
st.sidebar.title("Configuration")
model_option = st.sidebar.selectbox(
    "Choose a model:",
    ("llama-3.3-70b-versatile", "llama-3.1-8b-instant", "deepseek-r1-distill-llama-70b")
)

if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.rerun()

st.title("🌿 Plant Disease Assistant")
st.info("I can help identify issues like Apple Scab and suggest treatments.")

# 3. Initialize Groq Client
try:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
except Exception as e:
    st.error("Missing API Key! Please add it to Streamlit Secrets.")
    st.stop()

# 4. Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. Chat Logic
if prompt := st.chat_input("Ask about Apple Scab treatment..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # Call API with streaming
            stream = client.chat.completions.create(
                model=model_option,
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            )
            
            # Stream the response to UI and capture the full string
            full_response = st.write_stream(stream)
            
            # CRITICAL: Only save to history if the response isn't empty
            if full_response:
                st.session_state.messages.append({"role": "assistant", "content": full_response})
        
        except Exception as e:
            st.error(f"Error: {e}")
