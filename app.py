import streamlit as st
from groq import Groq

st.set_page_config(page_title="Simple Chat", page_icon="🤖")
st.title("🤖 Chat with AI")

# Initialize client
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# Initialize history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# User Input
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Response Generation
    with st.chat_message("assistant"):
        # We use a placeholder to update the text as it streams
        response_placeholder = st.empty()
        full_response = ""
        
        # Call the API
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
            stream=True
        )

        # Loop through the stream chunks
        for chunk in completion:
            # This extracts JUST the text content from the JSON
            content = chunk.choices[0].delta.content
            if content:
                full_response += content
                response_placeholder.markdown(full_response)

        # Save to history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
