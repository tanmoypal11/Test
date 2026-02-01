import streamlit as st
from groq import Groq

st.set_page_config(page_title="Gemma Chatbot", page_icon="🤖")
st.title("💬 Chat with Gemma")

# Access your API Key from Streamlit Secrets
# (We will set this up in the Cloud dashboard)
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model="gemma2-9b-it", # Or use "mistral-large-latest" for Mistral
            messages=st.session_state.messages,
            stream=True,
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})
