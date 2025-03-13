import streamlit as st
from pages import vietnamese, english

st.title("Text Summarization App")


option = st.sidebar.radio("Text language options:", ["Vietnamese", "English"])

if option == "Vietnamese":
    vietnamese.run()
elif option == "English":
    english.run()

