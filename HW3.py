import streamlit as st
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import anthropic

# Title of the Streamlit app
st.title("HW3 -- practicing Streamlit Code")

# Fetch the API keys from streamlit secrets
openai_api_key = st.secrets["OPEN_AI_KEY"]
claude_api_key = st.secrets["CLAUDE_AI_KEY"]

# Set API keys for OpenAI and Anthropic
client = OpenAI(api_key=st.secrets["OPEN_AI_KEY"])

# Sidebar elements
st.sidebar.title("Chat Bot Options")

# Conversation behavior options
behavior = st.sidebar.radio(
    "Conversation behavior:",
    (
        "Keep last 5 questions",
        "Summarize after 5 interactions",
        "Limit by token size (5000 tokens)",
    ),
)

# Model options for CHATBOT
selected_llm_for_chatbot = st.sidebar.selectbox(
    "Choose the model for Chatbot",
    (
        "OpenAI: gpt-3.5-turbo",
        "OpenAI: gpt-4 (Advanced)",
        "Claude: claude-3-haiku-20240307",
        "Claude: claude-3-5-sonnet-20240620 (Advanced)",
    ),
)

if selected_llm_for_chatbot == "OpenAI: gpt-3.5-turbo":
    model_to_use_for_chatbot = "gpt-3.5-turbo"

elif selected_llm_for_chatbot == "OpenAI: gpt-4 (Advanced)":
    model_to_use_for_chatbot = "gpt-4"

elif selected_llm_for_chatbot == "Claude: claude-3-haiku-20240307":
    model_to_use_for_chatbot = "claude-instant"

elif (
    selected_llm_for_chatbot == "Claude: claude-3-5-sonnet-20240620 (Advanced)"
):
    model_to_use_for_chatbot = "claude-2"
else:
    model_to_use_for_chatbot = None

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
        st.error(f"Failed to retrieve URL: {url}. Error: {e}")
        return ""

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "context_text" not in st.session_state:
    st.session_state["context_text"] = ""
if "urls" not in st.session_state:
    st.session_state["urls"] = {"url1": "", "url2": ""}

# Inputs for URLs
url1 = st.text_input("First URL:", value=st.session_state["urls"]["url1"])
url2 = st.text_input("Second URL:", value=st.session_state["urls"]["url2"])

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
        {"role": "system", "content": f"Here is some background information:\n{st.session_state['context_text']}"},
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
        {"role": "system", "content": f"Here is some background information:\n{st.session_state['context_text']}"},
        {"role": "assistant", "content": "How can I help you?"},
    ]

# Function to manage conversation memory
def manage_memory(messages, behavior):

    if behavior == "Keep last 5 questions":
        # Keep system messages and last 5 pairs
        system_messages = [msg for msg in messages if msg["role"] == "system"]
        conversation = [msg for msg in messages if msg["role"] != "system"]
        return system_messages + conversation[-10:]  # Last 5 pairs (user and assistant)

    elif behavior == "Summarize after 5 interactions":
        system_messages = [msg for msg in messages if msg["role"] == "system"]
        conversation = [msg for msg in messages if msg["role"] != "system"]

        if len(conversation) > 10:  # More than 5 pairs
            # Summarize the conversation
            document = "\n".join(
                [msg["content"] for msg in conversation if msg["role"] == "user"]
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
        
    elif behavior == "Limit by token size (5000 tokens)":
        system_messages = [msg for msg in messages if msg["role"] == "system"]
        conversation = [msg for msg in messages if msg["role"] != "system"]
        token_count = sum([len(msg["content"]) for msg in conversation])  # Rough estimation
        while token_count > 5000 and conversation:
            conversation.pop(0)  # Remove oldest messages until under the token limit
            token_count = sum([len(msg["content"]) for msg in conversation])
        return system_messages + conversation
    
    else:
        return messages

# Function to generate summary (needed for 'Summarize after 5 interactions')
def generate_summary(text, instruction, model_to_use):
    if model_to_use in ["gpt-3.5-turbo", "gpt-4"]:
        return summarize_with_openai(text, instruction, model_to_use)

    elif model_to_use.startswith("claude"):
        return summarize_with_claude(text, instruction, model_to_use)

    else:
        st.error("Model not supported.")
        return None

def summarize_with_openai(text, instruction, model):
    messages = [
        {"role": "system", "content": "You are a helpful assistant that summarizes conversations."},
        {"role": "user", "content": f"{instruction}\n\n{text}"},
    ]
    response = client.chat.completions.create(
        model=model, messages=messages, max_tokens=500
    )
    summary = response.choices[0].message.content
    return summary # commenting this because I am getting the summary twice

def summarize_with_claude(text, instruction, model):
    client = anthropic.Client(api_key=claude_api_key)
    prompt = f"{anthropic.HUMAN_PROMPT} {instruction}\n\n{text} {anthropic.AI_PROMPT}"
    response = client.completion(
        prompt=prompt,
        stop_sequences=[anthropic.HUMAN_PROMPT],
        model=model,
        max_tokens_to_sample=500,
    )
    return response["completion"]

# Manage conversation memory
st.session_state["messages"] = manage_memory(st.session_state["messages"], behavior)

# Display chat history
for msg in st.session_state["messages"]:
    if msg["role"] != "system":  # Skip the system messages
        chat_msg = st.chat_message(msg["role"])
        chat_msg.write(msg["content"])

# Capturing the user input for the chatbot
if prompt := st.chat_input("Ask the chatbot a question or interact:"):
    # Append the user's message to session state
    st.session_state["messages"].append({"role": "user", "content": prompt})

    # Display user's input in the chat
    with st.chat_message("user"):
        st.markdown(prompt)

    # Function to get chatbot response
    def get_chatbot_response(messages, model_to_use):
        if model_to_use in ["gpt-3.5-turbo", "gpt-4"]:
            return chatbot_response_openai(messages, model_to_use)
        elif model_to_use.startswith("claude"):
            return chatbot_response_claude(messages, model_to_use)
        else:
            st.error("Model not supported.")
            return None

    def chatbot_response_openai(messages, model):
        response = client.chat.completions.create(
            model=model, messages=messages
        )
        assistant_message = response.choices[0].message.content
        return assistant_message

    def chatbot_response_claude(messages, model):
        client = anthropic.Client(api_key=claude_api_key)
        prompt = ""
        for message in messages:
            if message["role"] == "system":
                prompt += f"{anthropic.HUMAN_PROMPT} {message['content']} {anthropic.AI_PROMPT}"
            elif message["role"] == "user":
                prompt += f"{anthropic.HUMAN_PROMPT} {message['content']} {anthropic.AI_PROMPT}"
            elif message["role"] == "assistant":
                prompt += f"{message['content']}"
        response = client.completion(
            prompt=prompt,
            stop_sequences=[anthropic.HUMAN_PROMPT],
            model=model,
            max_tokens_to_sample=500,
        )
        return response["completion"]

    # Get assistant's response
    assistant_message = get_chatbot_response(
        st.session_state["messages"], model_to_use_for_chatbot
    )

    # Append the assistant's response to session state
    st.session_state["messages"].append(
        {"role": "assistant", "content": assistant_message}
    )

    # Display assistant's response
    with st.chat_message("assistant"):
        st.markdown(assistant_message)