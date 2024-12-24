import os
from dotenv import load_dotenv
import streamlit as st
from groq import Groq
from huggingface_hub import InferenceClient

# Load environment variables from the .env file
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_TOKEN")

# Initialize session state variables if not already initialized
if "disabled" not in st.session_state:
    st.session_state.disabled = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Page Tab Title ---
st.set_page_config(page_icon=":crown:", layout="wide", page_title="ChatBot")

# --- Streamlit UI ---
st.title("Chat With Me")

# Model Dictionary
model_dict = {
    "Llama 3.3 70b": "llama-3.3-70b-versatile",
    "Gemma2 9b": "gemma2-9b-it",
    "Llama 3.2 90b Vision": "llama-3.2-90b-vision-preview",
    "Mixtral 8x7b": "mixtral-8x7b-32768"
}

# System Prompts
system_prompts_dict = {
    "General": "You are a helpful and friendly Computer Science tutor and coding assistant.  Your primary goal is to facilitate learning and provide coding support within the context of computer science.  You should be knowledgeable about various Computer Science domains including software development, web development, system administration, data science, information security, databases, networks, mobile development, cloud computing, DevOps, UX design, AI engineering, and related areas. Explain concepts clearly and provide practical examples.  Focus on coding languages like Java, Python, JavaScript, HTML, CSS and tech stacks that revolve around web development, such as MERN, MEAN, and other technologies like Next.js.  When asked to generate code, produce clean, well-commented, and efficient code in the requested language (if specified), or suggest the most suitable language given the task. Offer different implementation options where appropriate, along with explanations of the advantages and disadvantages of each approach. Consider best practices in software engineering principles such as modularity, readability, and maintainability. If asked about career paths or learning resources in Computer Science, provide up-to-date and helpful information based on the latest industry trends and educational opportunities.  Be encouraging and supportive, always aiming to empower the user to improve their computer science skills and knowledge.  Remember that learning is an iterative process and help the user work through challenges step by step.",

    "Code": "You are a precise and accurate coding assistant specializing in Java, Python, JavaScript, HTML, and CSS, as well as related web development technologies like MERN, MEAN, and Next.js. Your responses should be concise, technically correct, and adhere to best practices. When presented with coding questions or requests for code generation, provide efficient, well-commented code in the specified language (or suggest an appropriate language if none is given), emphasizing clarity and maintainability. If asked to explain code, provide detailed and insightful explanations. You are also proficient in other programming languages and computer science domains but prioritize those mentioned above.  Avoid giving general advice and stick strictly to the user's coding needs. If asked non-coding questions or opinions, say 'I'm a coding assistant and not qualified to respond to your inquiry. Feel free to provide a relevant coding question, and I will be very pleased to help!' If the query is related to finding resources to learn the mentioned tech stacks give details regarding the available courses, videos, books along with links if any.",
}

# Sidebar for Model and Task Selection
with st.sidebar:
    st.title("Parameters")
    temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.3, step=0.01)

    st.session_state.disabled = st.checkbox("Hyper  Coder", value=st.session_state.disabled)

    selected_model = st.selectbox(
        "Model Selection",
        model_dict.keys(),
        disabled=st.session_state.disabled
    )

    selected_task = st.selectbox(
        "Task",
        system_prompts_dict.keys(),
        disabled=st.session_state.disabled
    )

# Initialize Client Based on Toggle State
try:
    if st.session_state.disabled:
        client = InferenceClient(token=HUGGINGFACE_API_KEY)
    else:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
except Exception as e:
    st.error(f"Error initializing client: {e}")
    client = None

if st.session_state.disabled:
    model_name = "Qwen/Qwen2.5-Coder-32B-Instruct"
    selected_model = "Qwen 2.5 Coder 32B"
    system_prompt = system_prompts_dict.get("Code", None)
else:
    model_name = model_dict.get(selected_model, None)
    system_prompt = system_prompts_dict.get(selected_task, None)

st.markdown(f"Using Model: **{selected_model}**")

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
        if model_name and system_prompt and client:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    *[
                        {"role": "user", "content": m["content"]}
                        for m in st.session_state.chat_history if m["role"] == "user"
                    ],
                ],
                model=model_name,
                temperature=temperature,
                max_tokens=8192,
                top_p=0.9
            )
            bot_response = chat_completion.choices[0].message.content

            # Display Bot Response
            with st.chat_message("assistant"):
                st.markdown(bot_response)

            st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
        else:
            st.warning("Model or Task selection is disabled. Unable to process your input.")

    except Exception as e:
        st.error(f"Error processing your message: {e}")
