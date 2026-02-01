import streamlit as st
from groq import Groq

# 1. MUST BE FIRST: Set page config before any other st. commands
st.set_page_config(page_title="Gemma Chatbot", page_icon="🤖")

# 2. Sidebar for model selection
st.sidebar.title("Settings")
model_option = st.sidebar.selectbox(
    "Choose a model:",
    (
        "llama-3.1-8b-instant",      # Fast & Great for Chat
        "llama-3.3-70b-versatile",   # High Intelligence
        "deepseek-r1-distill-llama-70b" # Deep Reasoning
    )
)

st.title("💬 Chat with AI")
st.caption(f"Currently using: {model_option}")

# Access your API Key from Streamlit Secrets
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
        try:
            stream = client.chat.completions.create(
                model=model_option,
                messages=st.session_state.messages,
                stream=True,
            )
            response = st.write_stream(stream)
            
            # FIX: Only append if response is a valid string and not empty
            if response:
                st.session_state.messages.append({"role": "assistant", "content": str(response)})
            else:
                st.error("The model returned an empty response. Please try again.")
        
        except Exception as e:
            st.error(f"An error occurred: {e}")
