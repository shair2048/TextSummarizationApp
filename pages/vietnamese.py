import streamlit as st

def run():
    text = st.text_area("Enter Text Content:", height=200)
        
    if st.button("Summarize"):
        if text.strip():
            st.write("Summarization Complete")
        else:
            st.warning("Please enter text content.")

# if __name__ == "__main__":
#     main()