import os
from dotenv import load_dotenv
import streamlit as st
from groq import Groq

# Loading environment variables from the .env file
load_dotenv()

GROQ_API_KEY=os.getenv('GROQ_API_KEY')

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY) # Replace with your actual API key


# Model Configuration
model_name = "llama-3.2-90b-vision-preview"

# Initialize Chat History in Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Streamlit UI ---
st.title("Groq Chatbot")

# Sidebar for Temperature Slider
with st.sidebar:
    st.title("Parameters")
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.3, step=0.01)


# Display Chat History
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get User Input
user_input = st.chat_input("Type your message here...")

if user_input:
    # Display User Input
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Generate Response
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},  # Optional system message
                *([{"role": "user", "content": m["content"]} for m in st.session_state.chat_history if m["role"]=="user"]),
            ],
            model=model_name,
            temperature=temperature, # Use the slider value here.
            max_tokens=8192, 
        )
        bot_response = chat_completion.choices[0].message.content



        # Display Bot Response
        with st.chat_message("assistant"):
            st.markdown(bot_response)

        st.session_state.chat_history.append({"role": "assistant", "content": bot_response})

    except Exception as e:
        st.error(f"Error processing your message: {e}")