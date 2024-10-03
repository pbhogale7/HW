import streamlit as st
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import anthropic
import google.generativeai as genai
# Title of the Streamlit app
st.title("HW-03")


# Set API keys for OpenAI, Claude and Gemini
openai_client = OpenAI(api_key=st.secrets["OPEN_AI_KEY"])
claude_client = anthropic.Anthropic(api_key=st.secrets["CLAUDE_AI_KEY"])
genai.configure(api_key=st.secrets["GEMINI_AI_KEY"])

# Sidebar elements
st.sidebar.title("Chat Bot Options")

# Conversation behavior options
behavior = st.sidebar.radio(
    "Conversation behavior:",
    (
        "Buffer of 5 questions",
        "Conversation Summary",
        "Buffer of 5000 tokens",
    ),
)

# Model options for CHATBOT
selected_llm_for_chatbot = st.sidebar.selectbox(
    "Choose the model for Chatbot",
    ("OpenAI", "Claude", "Gemini")
)

# Simplified model selection
use_advanced_model = st.sidebar.checkbox("Use advanced model")

# Select the model version
if selected_llm_for_chatbot == "OpenAI":
    model_to_use_for_chatbot = "gpt-4" if use_advanced_model else "gpt-3.5-turbo"
elif selected_llm_for_chatbot == "Claude":
    model_to_use_for_chatbot = "claude-3-opus-20240229" if use_advanced_model else "claude-3-sonnet-20240229"
else:  # Gemini
    model_to_use_for_chatbot = "gemini-1.0-pro" if use_advanced_model else "gemini-1.0-pro-latest"

# Display the selected model
st.sidebar.write(f"Selected model: {model_to_use_for_chatbot}")

# Sidebar title
st.sidebar.title("URL Inputs")

# Sidebar text input
url1 = st.sidebar.text_input(
    "First URL:", value=st.session_state.get("urls", {}).get("url1", ""))
url2 = st.sidebar.text_input(
    "Second URL:", value=st.session_state.get("urls", {}).get("url2", ""))

# Function to extract text content from a URL


def extract_text_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Handle HTTP errors
        soup = BeautifulSoup(response.content, "html.parser")
        # Remove script and style elements
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
        return soup.get_text(separator="\n")
    except requests.RequestException as e:
        st.sidebar.error(f"Failed to retrieve URL: {url}. Error: {e}")
        return ""


# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "context_text" not in st.session_state:
    st.session_state["context_text"] = ""
if "urls" not in st.session_state:
    st.session_state["urls"] = {"url1": "", "url2": ""}

# Check if URLs have changed
if url1 != st.session_state["urls"]["url1"] or url2 != st.session_state["urls"]["url2"]:
    st.session_state["urls"]["url1"] = url1
    st.session_state["urls"]["url2"] = url2
    # Extract text and update context_text
    text1 = extract_text_from_url(url1) if url1 else ""
    text2 = extract_text_from_url(url2) if url2 else ""
    st.session_state["context_text"] = text1 + "\n" + text2
    # Reset the messages
    st.session_state["messages"] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "system",
            "content": "Your output should end with 'DO YOU WANT MORE INFO?' case sensitive unless you get a 'no' as an input. In which case, you should go back to asking 'How can I help you?' Make sure your answers are simple enough for a 10-year-old to understand.",
        },
        {"role": "system",
            "content": f"Here is some background information:\n{st.session_state['context_text']}"},
        {"role": "assistant", "content": "How can I help you?"},
    ]

# If no messages, initialize
if not st.session_state["messages"]:
    st.session_state["messages"] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "system",
            "content": "Your output should end with 'DO YOU WANT MORE INFO?' case sensitive unless you get a 'no' as an input. In which case, you should go back to asking 'How can I help you?' Make sure your answers are simple enough for a 10-year-old to understand.",
        },
        {"role": "system",
            "content": f"Here is some background information:\n{st.session_state['context_text']}"},
        {"role": "assistant", "content": "How can I help you?"},
    ]

# Function to manage conversation memory


def manage_memory(messages, behavior):

    if behavior == "Buffer of 5 questions":
        # Keep system messages and last 5 pairs
        system_messages = [msg for msg in messages if msg["role"] == "system"]
        conversation = [msg for msg in messages if msg["role"] != "system"]
        # Last 5 pairs (user and assistant)
        return system_messages + conversation[-10:]

    elif behavior == "Conversation Summary":
        system_messages = [msg for msg in messages if msg["role"] == "system"]
        conversation = [msg for msg in messages if msg["role"] != "system"]

        if len(conversation) > 10:  # More than 5 pairs
            # Summarize the conversation
            document = "\n".join(
                [msg["content"]
                    for msg in conversation if msg["role"] == "user"]
            )
            instruction = "Summarize this conversation."
            summary = generate_summary(
                document, instruction, model_to_use_for_chatbot
            )
            st.write("### Conversation Summary")
            st.write(summary)
            # Reset conversation keeping the system messages and summary
            return system_messages + [{"role": "assistant", "content": summary}]

        else:
            return messages

    elif behavior == "Buffer of 5000 tokens":
        system_messages = [msg for msg in messages if msg["role"] == "system"]
        conversation = [msg for msg in messages if msg["role"] != "system"]
        token_count = sum([len(msg["content"])
                          for msg in conversation])  # Rough estimation
        while token_count > 5000 and conversation:
            # Remove oldest messages until under the token limit
            conversation.pop(0)
            token_count = sum([len(msg["content"]) for msg in conversation])
        return system_messages + conversation

    else:
        return messages

# Function to generate summary (needed for 'Conversation Summary')


def generate_summary(text, instruction, model_to_use):
    if model_to_use in ["gpt-3.5-turbo", "gpt-4"]:
        return summarize_with_openai(text, instruction, model_to_use)

    elif model_to_use.startswith("claude"):
        return summarize_with_claude(text, instruction, model_to_use)

    else:
        st.error("Model not supported.")
        return None

# Function for conversation summary for openai


def summarize_with_openai(text, instruction, model):
    messages = [
        {"role": "system", "content": "You are a helpful assistant that summarizes conversations."},
        {"role": "user", "content": f"{instruction}\n\n{text}"},
    ]
    response = openai_client.chat.completions.create(
        model=model, messages=messages, max_tokens=500
    )
    summary = response.choices[0].message.content
    return summary  # commenting this because I am getting the summary twice

# Function for conversation summary for claude


def summarize_with_claude(text, instruction, model):
    prompt = f"{anthropic.HUMAN_PROMPT} {instruction}\n\n{text} {anthropic.AI_PROMPT}"
    response = claude_client.completion(
        prompt=prompt,
        stop_sequences=[anthropic.HUMAN_PROMPT],
        model=model,
        max_tokens_to_sample=500,
    )
    return response["completion"]


# Manage conversation memory
st.session_state["messages"] = manage_memory(
    st.session_state["messages"], behavior)

# Function to get chatbot response with streaming


def get_chatbot_response(messages, model_to_use):
    if model_to_use in ["gpt-3.5-turbo", "gpt-4"]:
        return openai_response_stream(messages, model_to_use)
    elif model_to_use.startswith("claude"):
        return claude_response_stream(messages, model_to_use)
    elif model_to_use.startswith("gemini"):

        return gemini_response_stream(messages, model_to_use)
    else:
        st.error("Model not supported.")
    return None


# Function for openai response stream
def openai_response_stream(messages, model):
    stream = openai_client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True
    )

    response = st.empty()
    full_response = ""
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            full_response += chunk.choices[0].delta.content
            response.markdown(full_response + "▌")
    response.markdown(full_response)
    return full_response


# Function for claude response stream
def claude_response_stream(messages, model):
    # Convert messages to Claude's format
    claude_messages = []
    system_message = ""

    for msg in messages:
        if msg["role"] == "system":
            system_message += msg["content"] + "\n"
        else:
            claude_messages.append(
                {
                    "role": "user" if msg["role"] == "user" else "assistant",
                    "content": msg["content"]
                }
            )

    # Prepend system message to the first user message or create a user message if claude_messages is empty
    if system_message.strip():
        if claude_messages and claude_messages[0]["role"] == "user":
            claude_messages[0]["content"] = system_message + \
                claude_messages[0]["content"]
        else:
            claude_messages.insert(
                0, {"role": "user", "content": system_message})

    try:
        response = st.empty()
        full_response = ""

        with claude_client.messages.stream(
            model=model,
            messages=claude_messages,
            max_tokens=1024
        ) as stream:
            for chunk in stream:
                if chunk.type == "content_block_delta":
                    full_response += chunk.delta.text
                    response.markdown(full_response + "▌")

        response.markdown(full_response)
        return full_response
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None

# Function for gemini response stream


def gemini_response_stream(messages, model):
    # Convert messages to Gemini's format
    gemini_messages = []
    system_message = ""

    for msg in messages:
        if msg["role"] == "system":
            system_message += msg["content"] + "\n"
        elif msg["role"] == "user":
            gemini_messages.append(
                {"role": "user", "parts": [{"text": msg["content"]}]})
        elif msg["role"] == "assistant":
            gemini_messages.append(
                {"role": "model", "parts": [{"text": msg["content"]}]})

    # Prepend system message to the first user message or create a user message if gemini_messages is empty
    if system_message.strip():
        if gemini_messages and gemini_messages[0]["role"] == "user":
            gemini_messages[0]["parts"][0]["text"] = system_message + \
                gemini_messages[0]["parts"][0]["text"]
        else:
            gemini_messages.insert(
                0, {"role": "user", "parts": [{"text": system_message}]})

    try:
        model = genai.GenerativeModel(model_to_use_for_chatbot)
        chat = model.start_chat(history=gemini_messages)

        response = st.empty()
        full_response = ""

        for chunk in chat.send_message("Continue the conversation", stream=True):
            full_response += chunk.text
            response.markdown(full_response + "▌")

        response.markdown(full_response)
        return full_response
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None


# Check if at least one URL has been entered
if url1 or url2:
    # Display chat history
    for msg in st.session_state["messages"]:
        if msg["role"] != "system":  # Skip the system messages
            chat_msg = st.chat_message(msg["role"])
            chat_msg.write(msg["content"])

    # Capturing the user input for the chatbot
    if prompt := st.chat_input("Ask the chatbot a question or interact:"):
        # Append the user's message to session state
        st.session_state["messages"].append(
            {"role": "user", "content": prompt})

        # Display user's input in the chat
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get assistant's response with streaming
        with st.chat_message("assistant"):
            assistant_message = get_chatbot_response(
                st.session_state["messages"], model_to_use_for_chatbot
            )

        # Append the assistant's response to session state
        if assistant_message:
            # Append the assistant's response to session state
            st.session_state["messages"].append(
                {"role": "assistant", "content": assistant_message}
            )
else:
    st.info("Please enter at least one URL in the sidebar to start the chat.")