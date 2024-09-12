import streamlit as st
st.set_page_config(page_title= "HW -IST 688")

# Show title and description.
st.title("HW Manager")

# Ask user for their OpenAI API key via st.text_input.
# Alternatively, you can store the API key in ./.streamlit/secrets.toml and access it
# via st.secrets, see https://docs.streamlit.io/develop/concepts/connections/secrets-management

hw1_page = st.Page("HW1.py", title="HW1")
hw2_page = st.Page("HW2.py", title="HW2" , default=True)

pg = st.navigation([hw1_page, hw2_page])

pg.run()
