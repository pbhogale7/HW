from bs4 import BeautifulSoup
import requests
import streamlit as st
from openai import OpenAI

st.title("This is HW 2")


# Ask user for their OpenAI API key via st.text_input.
# Alternatively, you can store the API key in ./.streamlit/secrets.toml and access it
# via st.secrets, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
##openai_api_key = st.text_input("OpenAI API Key", type="password")

openai_api_key = st.secrets["OPEN_AI_KEY"] 
#if not openai_api_key: 
#    st.info("Please add your OpenAI API key to continue.", icon="🗝")
#else:

#    # Create an OpenAI client.
#    client = OpenAI(api_key=openai_api_key)

    # Let the user upload a file via st.file_uploader.
 #   uploaded_file = st.file_uploader(
#        "Upload a document (.txt or .md)", type=("txt", "md")
 #   )

    # Ask the user for a question via st.text_area.
 #   question = st.text_area(
  #      "Now ask a question about the document!",
  #      placeholder="Can you give me a short summary?",
  #      disabled=not uploaded_file,
 #   )

  #  if uploaded_file and question:

        # Process the uploaded file and question.
 #       document = uploaded_file.read().decode()
 #       messages = [
  #          {
  #              "role": "user",
 #               "content": f"Here's a document: {document} \n\n---\n\n {question}",
  #          }
  #      ]

        # Generate an answer using the OpenAI API.
  #      stream = client.chat.completions.create(
   #         model="gpt-4o-mini",
   #         messages=messages,
   #         stream=True,
   #     )

        # Stream the response to the app using st.write_stream.
   #     st.write_stream(stream)

def read_url_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP errors
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except requests.RequestException as e:
        st.error(f"Error reading {url}: {e}")
        return None

# Create an OpenAI client.
client = OpenAI(api_key=openai_api_key)  

url = st.text_input("Enter a URL")

    # Sidebar for selecting summary type (similar to your previous Lab2)
summary_type = st.selectbox("Select Summary Type", ["Summarize the document in 100 words", "Summarize the document in 2 connecting paragraphs", "Summarize the document in 5 bullet points"])

    # Step 8: Dropdown menu to select output language
language = st.selectbox("Select Output Language", ["English", "French", "Spanish"])

    # Step 10: Option to select LLM models
llm_model = st.sidebar.selectbox("Select LLM", ["OpenAI", "Claude", "Cohere"])


    # Step 6: Display summary
if st.button("Summarize"):
        if url:
            content = read_url_content(url)
            if content and summary_type and language:
                messages_openai = [
            {
                "role": "user",
                "content": f"Here's a document: {content} \n\n---\n\n {summary_type} in {language}",
            }
        ]
                if llm_model == "OpenAI":
                    stream = client.chat.completions.create(
                         model="gpt-4o-mini",
                         messages=messages_openai,
                         stream=True,)
                    st.write("Open AI's Response:")
                    st.write_stream(stream)
                elif llm_model =="Claude":
               #Enter code for Claude using Claude Syntax.
                    st.write("Claude's  Response:")
                elif llm_model == "Cohere":
			#Enter code for Cohere using Cohere Syntax.
                    st.write("Cohere's  Response:")

        else:
            st.error("Please enter a valid URL.")