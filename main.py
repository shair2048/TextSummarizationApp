import streamlit as st
from pages import vietnamese, english

option = st.sidebar.radio("Options:", ["Vietnamese", "English"])

if option == "Vietnamese":
    vietnamese.run()
elif option == "English":
    english.run()

