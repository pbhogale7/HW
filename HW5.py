import sys
import streamlit as st
from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai
from bs4 import BeautifulSoup
import os
import zipfile
import tempfile
from collections import deque
import numpy as np

# Workaround for sqlite3 issue in Streamlit Cloud
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb

st.title("HW 5")

# Function to ensure the OpenAI client is initialized
def ensure_openai_client():
    if 'openai_client' not in st.session_state:
        api_key = st.secrets["OPEN_AI_KEY"]
        st.session_state.openai_client = OpenAI(api_key=api_key)

# Function to ensure the Anthropic client is initialized
def ensure_anthropic_client():
    if 'anthropic_client' not in st.session_state:
        api_key = st.secrets["CLAUDE_AI_KEY"]
        st.session_state.anthropic_client = Anthropic(api_key=api_key)

# Function to ensure the Google AI client is initialized
def ensure_google_ai_client():
    if 'google_ai_client' not in st.session_state:
        api_key = st.secrets["GEMINI_AI_KEY"]
        genai.configure(api_key=api_key)
        st.session_state.google_ai_client = genai

# Function to extract HTML files from zip
def extract_html_from_zip(zip_path):
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        html_files = {}
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                if file.endswith('.html'):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        html_files[file] = f.read()
    return html_files

# Function to create the ChromaDB collection
def create_hw4_collection():
    if 'HW_URL_Collection' not in st.session_state:
        persist_directory = os.path.join(os.getcwd(), "chroma_db")
        client = chromadb.PersistentClient(path=persist_directory)
        collection = client.get_or_create_collection("HW_URL_Collection")

        zip_path = os.path.join(os.getcwd(), "su_orgs.zip")
        if not os.path.exists(zip_path):
            st.error(f"Zip file not found: {zip_path}")
            return None

        html_files = extract_html_from_zip(zip_path)

        if collection.count() == 0:
            with st.spinner("Processing content and preparing the system..."):
                ensure_openai_client()

                for filename, content in html_files.items():
                    try:
                        soup = BeautifulSoup(content, 'html.parser')
                        text = soup.get_text(separator=' ', strip=True)

                        response = st.session_state.openai_client.embeddings.create(
                            input=text, model="text-embedding-3-small"
                        )
                        embedding = response.data[0].embedding

                        collection.add(
                            documents=[text],
                            metadatas=[{"filename": filename}],
                            ids=[filename],
                            embeddings=[embedding]
                        )
                    except Exception as e:
                        st.error(f"Error processing {filename}: {str(e)}")
        else:
            st.info("Using existing vector database.")

        st.session_state.HW_URL_Collection = collection

    return st.session_state.HW_URL_Collection

# Function to get relevant club info based on the query
def get_relevant_info(query, model):
    collection = st.session_state.HW_URL_Collection
    
    # Always use OpenAI for embeddings, regardless of the selected model
    ensure_openai_client()
    try:
        response = st.session_state.openai_client.embeddings.create(
            input=query, model="text-embedding-3-small"
        )
        query_embedding = response.data[0].embedding
    except Exception as e:
        st.error(f"Error creating OpenAI embedding: {str(e)}")
        return "", []

    # Normalize the embedding
    query_embedding = np.array(query_embedding) / np.linalg.norm(query_embedding)

    try:
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=3
        )
        relevant_texts = results['documents'][0]
        relevant_docs = [result['filename'] for result in results['metadatas'][0]]
        return "\n".join(relevant_texts), relevant_docs
    except Exception as e:
        st.error(f"Error querying the database: {str(e)}")
        return "", []

# Function to get chatbot response using the selected LLM
def get_chatbot_response(query, context, conversation_memory, model):
    system_message = "You are an AI assistant with knowledge from specific documents. Use the provided context to answer the user's questions. If the information is not in the context, say you don't know based on the available information. Maintain consistency with your previous answers."

    condensed_history = "\n".join(
        [f"Human: {exchange['question']}\nAI: {exchange['answer']}" for exchange in conversation_memory])

    if model == "OpenAI GPT-4":
        ensure_openai_client()
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Context: {context}\n\nConversation history:\n{condensed_history}\n\nQuestion: {query}"}
        ]
        try:
            response = st.session_state.openai_client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                stream=True
            )
            return response
        except Exception as e:
            st.error(f"Error getting GPT-4 response: {str(e)}")
            return None

    elif model == "Anthropic Claude":
        ensure_anthropic_client()
        messages = [
            {"role": "user", "content": f"Here's some context information: {context}\n\nConversation history:\n{condensed_history}"},
            {"role": "assistant", "content": "I understand. I'll use this context and conversation history to answer questions consistently. What would you like to know?"},
            {"role": "user", "content": query}
        ]
        try:
            response = st.session_state.anthropic_client.messages.create(
                model="claude-3-opus-20240229",
                system=system_message,
                messages=messages,
                max_tokens=1024,
                stream=True
            )
            return response
        except Exception as e:
            st.error(f"Error getting Claude response: {str(e)}")
            return None

    elif model == "Google Gemini":
        ensure_google_ai_client()
        prompt = f"{system_message}\n\nContext: {context}\n\nConversation history:\n{condensed_history}\n\nHuman: {query}\nAI:"
        try:
            model = st.session_state.google_ai_client.GenerativeModel('gemini-1.0-pro')
            response = model.generate_content(prompt, stream=True)
            return response
        except Exception as e:
            st.error(f"Error getting Gemini response: {str(e)}")
        return None

# The main logic of the application is now at the global scope
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'conversation_memory' not in st.session_state:
    st.session_state.conversation_memory = deque(maxlen=5)
if 'system_ready' not in st.session_state:
    st.session_state.system_ready = False
if 'collection' not in st.session_state:
    st.session_state.collection = None

st.sidebar.title("Model Selection")
selected_model = st.sidebar.radio(
    "Choose an LLM:", ("OpenAI GPT-4", "Anthropic Claude", "Google Gemini"))

st.title("Chatbot Assistant for iSchool")

if not st.session_state.system_ready:
    with st.spinner("Processing documents and preparing the system..."):
        st.session_state.collection = create_hw4_collection()
        if st.session_state.collection:
            st.session_state.system_ready = True
            st.success("AI ChatBot is Ready!")
        else:
            st.error("Failed to create or load the document collection. Please check the zip file and try again.")

if st.session_state.system_ready and st.session_state.collection:
    st.subheader(f"Chat with the AI Assistant (Using {selected_model})")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Ask a question about the documents:")

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)

        relevant_texts, relevant_docs = get_relevant_info(user_input, selected_model)

        response_stream = get_chatbot_response(
            user_input, relevant_texts, st.session_state.conversation_memory, selected_model)

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            if selected_model == "OpenAI GPT-4":
                for chunk in response_stream:
                    if chunk.choices[0].delta.content is not None:
                        full_response += chunk.choices[0].delta.content
                        response_placeholder.markdown(full_response + "▌")
            elif selected_model == "Anthropic Claude":
                for chunk in response_stream:
                    chunk_type = getattr(chunk, 'type', None)
                    if chunk_type == 'content_block_delta':
                        if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'text'):
                            full_response += chunk.delta.text
                            response_placeholder.markdown(full_response + "▌")
                    elif chunk_type == 'message_delta':
                        if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'content'):
                            for content_block in chunk.delta.content:
                                if content_block.type == 'text':
                                    full_response += content_block.text
                                    response_placeholder.markdown(full_response + "▌")
            elif selected_model == "Google Gemini":
                for chunk in response_stream:
                    full_response += chunk.text
                    response_placeholder.markdown(full_response + "▌")
            response_placeholder.markdown(full_response)

        st.session_state.chat_history.append(
            {"role": "user", "content": user_input})
        st.session_state.chat_history.append(
            {"role": "assistant", "content": full_response})

        st.session_state.conversation_memory.append({
            "question": user_input,
            "answer": full_response
        })

        with st.expander("Relevant documents used"):
            for doc in relevant_docs:
                st.write(f"- {doc}")

elif not st.session_state.system_ready:
    st.info("The system is still preparing. Please wait...")
else:
    st.error("Failed to create or load the document collection. Please check the zip file and try again.")