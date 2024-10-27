import streamlit as st
st.set_page_config(page_title= "HW -IST 688")

# Show title and description.
st.title("HW Manager")

# Ask user for their OpenAI API key via st.text_input.
# Alternatively, you can store the API key in ./.streamlit/secrets.toml and access it
# via st.secrets, see https://docs.streamlit.io/develop/concepts/connections/secrets-management

hw1_page = st.Page("HW1.py", title="HW1")
hw2_page = st.Page("HW2.py", title="HW2")
hw3_page = st.Page("HW3.py", title="HW3")
hw5_page = st.Page("HW5.py", title="HW5", default= True)
audio_page = st.Page("presentation.py", title = "AUDIO")
pg = st.navigation([audio_page, hw1_page, hw2_page, hw3_page, hw5_page])

pg.run()
