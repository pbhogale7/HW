import streamlit as st
from openai import OpenAI
import pandas as pd
from datetime import datetime, timedelta
from collections import deque
import os
import sys

# SQLite adaptation for Streamlit Cloud
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb
from chromadb.config import Settings

# Application title and intro
st.title("LexiBot: Your Legal News Companion")
st.write("Your AI-powered tool for recent updates in the legal world.")

# Initialize session state if required
if 'dialog_history' not in st.session_state:
    st.session_state.dialog_history = []
if 'memory_queue' not in st.session_state:
    st.session_state.memory_queue = deque(maxlen=5)
if 'system_ready' not in st.session_state:
    st.session_state.system_ready = False
if 'openai_conn' not in st.session_state:
    api_key = st.secrets["OPEN_AI_KEY"]
    st.session_state.openai_conn = OpenAI(api_key=api_key)
if 'prev_query_type' not in st.session_state:
    st.session_state.prev_query_type = None

# Configure ChromaDB client
if 'chroma_client' not in st.session_state:
    try:
        db_path = os.path.join(os.getcwd(), "chroma_db")
        st.session_state.chroma_client = chromadb.PersistentClient(path=db_path)
        st.session_state.collection_ref = st.session_state.chroma_client.get_or_create_collection(name="legal_articles")
        st.success("ChromaDB initialized and ready!")
    except Exception as e:
        st.error(f"Could not set up ChromaDB: {e}")
        st.stop()

# Function to load legal article data into ChromaDB
def load_article_data(csv_file, collection):
    if not os.path.exists(csv_file):
        st.error("CSV file not found.")
        st.stop()
    
    data = pd.read_csv(csv_file)
    for idx, entry in data.iterrows():
        try:
            pub_date = (datetime(2000, 1, 1) + timedelta(days=int(entry['days_since_2000']))).strftime('%Y-%m-%d')
            article_content = f"{entry['company_name']} on {pub_date}: {entry['Document']} ({entry['URL']})"
            
            # Generate embedding with OpenAI
            embedding = generate_openai_embedding(article_content)
            collection.add(
                documents=[article_content],
                embeddings=[embedding],
                metadatas=[{
                    "company": entry['company_name'],
                    "date": pub_date,
                    "url": entry['URL']
                }],
                ids=[f"doc_{idx}"]
            )
        except Exception as e:
            st.error(f"Error processing entry {idx}: {e}")

# Helper function for embeddings generation
def generate_openai_embedding(text):
    st.info("🔄 Contacting OpenAI for embedding generation...")
    response = st.session_state.openai_conn.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

# Function to determine query type (chat or news-related)
def assess_query_type(query):
    # Handle as news if previous query was news-related and 'expand' keyword is present
    if st.session_state.prev_query_type == "news-related" and "expand" in query.lower():
        return "news-related"

    classification_prompt = [
        {"role": "system", "content": "You are a bot that categorizes questions."},
        {"role": "user", "content": f"Classify this query as either 'chat' or 'news-related': '{query}'"}
    ]
    response = st.session_state.openai_conn.chat.completions.create(
        model="gpt-4o",
        messages=classification_prompt,
        temperature=0
    )
    classification = response.choices[0].message.content.strip().lower()
    
    if "news-related" in classification:
        return "news-related"
    elif "chat" in classification:
        return "chat"
    else:
        return "unknown"

# Load data if collection is empty
csv_file_path = os.path.join(os.getcwd(), "Example_news_info_for_testing.csv")
if st.session_state.collection_ref.count() == 0:
    st.info("Loading articles into database...")
    load_article_data(csv_file_path, st.session_state.collection_ref)
    st.success("Article data loaded!")
else:
    st.info("Using existing collection data.")

st.session_state.system_ready = True

# Chat interface display
st.subheader("Ask the Legal Insights Assistant")

# Show conversation history
for message in st.session_state.dialog_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Capture user input
user_query = st.chat_input("Enter your legal news inquiry (e.g., 'updates on recent regulations'): ")

if user_query:
    with st.chat_message("user"):
        st.markdown(user_query)

    try:
        # Determine query type with the model
        query_type = assess_query_type(user_query)

        if query_type == "chat":
            # Respond with a conversational message
            response_text = "I'm ready to assist with your legal news inquiries!"
            with st.chat_message("assistant"):
                st.markdown(response_text)

            # Add response to history
            st.session_state.dialog_history.append({"role": "user", "content": user_query})
            st.session_state.dialog_history.append({"role": "assistant", "content": response_text})
            st.session_state.prev_query_type = "chat"
        
        elif query_type == "news-related":
            st.info("🔄 Generating embedding for your query...")

            # Create query embedding
            query_embedding = generate_openai_embedding(user_query)
            
            # Search ChromaDB
            st.subheader("Database Lookup")
            st.info("🔍 Searching the database for relevant articles...")
            results = st.session_state.collection_ref.query(
                query_embeddings=[query_embedding],
                n_results=5
            )
            
            matching_articles = results['documents'][0]
            article_metadata = [f"{meta['company']} - {meta['date']}" for meta in results['metadatas'][0]]
            
            # Display related articles
            st.subheader("Relevant Articles")
            if len(article_metadata) > 0:
                st.info(f"📰 Found {len(article_metadata)} related articles:")
                for detail in article_metadata:
                    st.write(f"📄 {detail}")
            else:
                st.warning("No matching articles found.")
            
            # LLM input preparation
            instructions = """You're an AI assistant focused on analyzing legal news for a global law firm. Prioritize legal impacts, regulatory changes, business effects, and precedent-setting cases.

            For each article, provide:
            - Summary of the event
            - Key legal points
            - Relevance to business
            - Importance to law firm operations."""
            
            # Build chat history
            conversation_context = "\n".join([
                f"User: {entry['question']}\nAssistant: {entry['answer']}" 
                for entry in st.session_state.memory_queue
            ])
            
            # Construct message for AI
            messages = [
                {"role": "system", "content": instructions},
                {"role": "user", "content": f"""Articles: {' '.join(matching_articles)}

Conversation history:
{conversation_context}

Query: {user_query}"""
                }
            ]
            
            # Stream AI response
            st.write("🤔 Preparing response...")
            response = st.session_state.openai_conn.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.7,
                stream=True
            )
            
            # Stream AI output in real-time
            with st.chat_message("assistant"):
                response_holder = st.empty()
                full_response = ""
                
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        response_holder.markdown(full_response)
            
            # Update history
            st.session_state.dialog_history.append({"role": "user", "content": user_query})
            st.session_state.dialog_history.append({"role": "assistant", "content": full_response})
            st.session_state.memory_queue.append({
                "question": user_query,
                "answer": full_response
            })
            st.session_state.prev_query_type = "news-related"
            
            # Expandable section for article references
            with st.expander("📄 View Article Sources"):
                for detail in article_metadata:
                    st.write(f"- {detail}")
        
        else:
            response_text = "I couldn't classify your query. Try rephrasing your question."
            with st.chat_message("assistant"):
                st.markdown(response_text)
    
    except Exception as e:
        st.error(f"An error occurred: {e}")
