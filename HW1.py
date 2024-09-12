import streamlit as st
from openai import OpenAI
import fitz # PyMuPDF
import io
# Function to read PDF file
def read_pdf(file):
 
    pdf_document = fitz.open(stream=file.read(), filetype="pdf")
 
    text = ""
 
    for page_num in range(pdf_document.page_count):
 
        page = pdf_document.load_page(page_num)
 
        text += page.get_text("text")
 
    return text
# Show title and description.
st.title("Poonam Bhogale - Document Question Answering")
st.write(
 
    "Upload a document below and ask a question about it – GPT will answer! "
 
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
)
# Ask user for their OpenAI API key via st.text_input.
openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
 
    st.info("Please add your OpenAI API key to continue.")
else:
 
    # Create an OpenAI client.
 
    client = OpenAI(api_key=openai_api_key)
 
    # Let the user upload a file via st.file_uploader.
 
    uploaded_file = st.file_uploader("Upload a document (.pdf or .txt)", type=("pdf", "txt"))
 
    # Initialize a flag to check if file was uploaded and processed
 
    file_processed = False
 
    if uploaded_file:

        # Check file extension
    
        file_extension = uploaded_file.name.split('.')[-1]
    
        if file_extension == 'txt':
    
            document = uploaded_file.read().decode()
    
            file_processed = True
    
    elif file_extension == 'pdf':
    
            document = read_pdf(uploaded_file)
    
            file_processed = True
    
    else:
    
            st.error("Unsupported file type.")
    
    if file_processed:
    
        # Ask the user for a question via st.text_area.
    
        question = st.text_area(
    
            "Now ask a question about the document!",
    
            placeholder="Can you give me a short summary?",
    
            disabled=not uploaded_file,
    
    )
    
    if question:
    
        # Process the uploaded file and question.
    
        messages = [
    
    {
    
            "role": "user",
            "content": f"Here's a document: {document} \n\n---\n\n {question}",
    
    }
    
    ]
    
    # Generate an answer using the OpenAI API.
    
    stream = client.chat.completions.create(
    
        model="gpt-4o-mini",
    
        messages=messages,
    
        stream=True,
    
    )
    
        # Stream the response to the app using st.write_stream.
    
    st.write_stream(stream)
    
    # If the file is removed, clear data
    
    if not uploaded_file:
    
        document = None
    
        file_processed = False
    
        st.info("File has been removed. No data is being used from the file.")