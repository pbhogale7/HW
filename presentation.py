import sys
import streamlit as st
from openai import OpenAI
from bs4 import BeautifulSoup
import os
import zipfile
import tempfile
from collections import deque
import numpy as np
from audio_recorder_streamlit import audio_recorder
import base64
import time

# Workaround for sqlite3 issue
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb

# Initialize OpenAI client
def setup_openai_client():
    if 'openai_client' not in st.session_state:
        api_key = st.secrets["openai_api_key"]
        st.session_state.openai_client = OpenAI(api_key=api_key)
    return st.session_state.openai_client

def cleanup_audio_files():
    """Clean up any temporary audio files"""
    try:
        for file in os.listdir():
            if (file.startswith("audio_input_") or file.startswith("audio_response_")) and file.endswith(".mp3"):
                try:
                    os.remove(file)
                except Exception as e:
                    st.warning(f"Could not remove audio file {file}: {str(e)}")
    except Exception as e:
        st.warning(f"Error during audio cleanup: {str(e)}")

# Audio Processing Functions
def transcribe_audio(client, audio_path):
    with open(audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
        return transcript.text

def text_to_audio(client, text, audio_path):
    response = client.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=text
    )
    response.stream_to_file(audio_path)

def auto_play_audio(audio_file):
    if os.path.exists(audio_file):
        with open(audio_file, "rb") as f:
            audio_bytes = f.read()
        base64_audio = base64.b64encode(audio_bytes).decode("utf-8")
        audio_html = f'<audio src="data:audio/mp3;base64,{base64_audio}" controls autoplay>'
        st.markdown(audio_html, unsafe_allow_html=True)

# ChromaDB Functions
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

def create_su_club_collection():
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
                client = setup_openai_client()

                for filename, content in html_files.items():
                    try:
                        soup = BeautifulSoup(content, 'html.parser')
                        text = soup.get_text(separator=' ', strip=True)

                        response = client.embeddings.create(
                            input=text,
                            model="text-embedding-3-small"
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

        st.session_state.HW_URL_Collection = collection

    return st.session_state.HW_URL_Collection

def get_relevant_info(query):
    collection = st.session_state.HW_URL_Collection
    client = setup_openai_client()

    try:
        response = client.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        )
        query_embedding = response.data[0].embedding
    except Exception as e:
        st.error(f"Error creating OpenAI embedding: {str(e)}")
        return "", []

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

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'last_recorded_audio' not in st.session_state:
    st.session_state.last_recorded_audio = None
if 'awaiting_response' not in st.session_state:
    st.session_state.awaiting_response = False
if 'temp_message' not in st.session_state:
    st.session_state.temp_message = None
if 'conversation_memory' not in st.session_state:
    st.session_state.conversation_memory = deque(maxlen=5)
if 'system_ready' not in st.session_state:
    st.session_state.system_ready = False
if 'cleanup_on_start' not in st.session_state:
    cleanup_audio_files()
    st.session_state.cleanup_on_start = True

st.title("iSchool Voice-Enabled RAG Chatbot")

# Initialize ChromaDB
if not st.session_state.system_ready:
    with st.spinner("Processing documents and preparing the system..."):
        st.session_state.collection = create_su_club_collection()
        if st.session_state.collection:
            st.session_state.system_ready = True
            st.success("AI ChatBot is Ready!")

# Display chat history
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        st.write(message["content"])

# Text and voice input
col1, col2 = st.columns([8, 2])
with col2:
    st.markdown('<div class="voice-recorder-container">', unsafe_allow_html=True)
    recorded_audio = audio_recorder(text="", recording_color="#e74c3c", neutral_color="#95a5a6", key="voice_recorder")
    st.markdown('</div>', unsafe_allow_html=True)

with col1:
    text_input = st.text_input("Type your message or use voice input...")

# Process text input
if text_input:
    st.session_state.messages.append({"role": "user", "content": text_input})
    # Fetch relevant information, process message, and display response
    relevant_texts, relevant_docs = get_relevant_info(text_input)
    response, tool_usage_info, _ = process_message(text_input, relevant_texts, st.session_state.conversation_memory)
    st.session_state.messages.append({"role": "assistant", "content": response})

# Process audio input if recorded
if recorded_audio and recorded_audio != st.session_state.last_recorded_audio:
    st.session_state.last_recorded_audio = recorded_audio
    audio_file = f"audio_input_{int(time.time())}.mp3"
    with open(audio_file, "wb") as f:
        f.write(recorded_audio)

    transcribed_text = transcribe_audio(setup_openai_client(), audio_file)
    st.session_state.messages.append({"role": "user", "content": f"🎤 {transcribed_text}"})

    relevant_texts, relevant_docs = get_relevant_info(transcribed_text)
    response, tool_usage_info, response_audio = process_message(transcribed_text, relevant_texts, st.session_state.conversation_memory, is_voice=True)
    st.session_state.messages.append({"role": "assistant", "content": response})

    os.remove(audio_file)
    if response_audio:
        auto_play_audio(response_audio)
