import os
from dotenv import load_dotenv
import streamlit as st
from groq import Groq

# Loading environment variables from the .env file
load_dotenv()

GROQ_API_KEY=os.getenv('GROQ_API_KEY')

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY) # Replace with your actual API key


# Initialize Chat History in Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Page Tab Title ---
st.set_page_config(page_icon=":crown:", layout="wide", page_title="ChatBot")

# --- Streamlit UI ---
st.title("Chat With Me")

#Model Dictionary
model_dict = {
    "Llama 3.3 70b" : "llama-3.3-70b-versatile",
    "Gemma2 9b" : "gemma2-9b-it",
    "Llama 3.2 90b Vision" : "llama-3.2-90b-vision-preview",
    "Mixtral 8x7b" : "mixtral-8x7b-32768"
}

# Model Selection
available_models = ("Llama 3.3 70b", "Gemma2 9b", "Llama 3.2 90b Vision", "Mixtral 8x7b")

# System Prompts
system_prompts_dict = {
    "General" : "You are a helpful and friendly Computer Science tutor and coding assistant.  Your primary goal is to facilitate learning and provide coding support within the context of computer science.  You should be knowledgeable about various Computer Science domains including software development, web development, system administration, data science, information security, databases, networks, mobile development, cloud computing, DevOps, UX design, AI engineering, and related areas. Explain concepts clearly and provide practical examples.  Focus on coding languages like Java, Python, JavaScript, HTML, CSS and tech stacks that revolve around web development, such as MERN, MEAN, and other technologies like Next.js.  When asked to generate code, produce clean, well-commented, and efficient code in the requested language (if specified), or suggest the most suitable language given the task. Offer different implementation options where appropriate, along with explanations of the advantages and disadvantages of each approach. Consider best practices in software engineering principles such as modularity, readability, and maintainability. If asked about career paths or learning resources in Computer Science, provide up-to-date and helpful information based on the latest industry trends and educational opportunities.  Be encouraging and supportive, always aiming to empower the user to improve their computer science skills and knowledge.  Remember that learning is an iterative process and help the user work through challenges step by step.",
    "Code" : "You are a precise and accurate coding assistant specializing in Java, Python, JavaScript, HTML, and CSS, as well as related web development technologies like MERN, MEAN, and Next.js. Your responses should be concise, technically correct, and adhere to best practices. When presented with coding questions or requests for code generation, provide efficient, well-commented code in the specified language (or suggest an appropriate language if none is given), emphasizing clarity and maintainability. If asked to explain code, provide detailed and insightful explanations. You are also proficient in other programming languages and computer science domains but prioritize those mentioned above.  Avoid giving general advice and stick strictly to the user's coding needs. If asked non-coding questions or opinions, say 'I'm a coding assistant and not qualified to respond to your inquiry. Feel free to provide a relevant coding question, and I will be very pleased to help!' If the query is related to finding resources to learn the mentioned tech stacks give details regarding the available courses, videos, books along with links if any.",
}

# Task Selection
available_tasks = ("General", "Code")

# Sidebar for Temperature Slider
with st.sidebar:
    st.title("Parameters")

    temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.3, step=0.01)

    selected_model = st.selectbox(
    "Model Selection",
    available_models)

    selected_task = st.selectbox(
        "Task",
        available_tasks,
    )

model_name = model_dict[selected_model]
system_prompt = system_prompts_dict[selected_task]

st.markdown(f"Using Model: **{selected_model}**")
st.markdown(f"Selected Tasks: **{selected_task}**")


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
                {"role": "system", "content": "You are a helpful assistant."},
                *([{"role": "user", "content": m["content"]} for m in st.session_state.chat_history if m["role"]=="user"]),
            ],
            model=model_name,
            temperature=temperature,
            max_tokens=32768,
        )
        bot_response = chat_completion.choices[0].message.content



        # Display Bot Response
        with st.chat_message("assistant"):
            st.markdown(bot_response)

        st.session_state.chat_history.append({"role": "assistant", "content": bot_response})

    except Exception as e:
        st.error(f"Error processing your message: {e}")
