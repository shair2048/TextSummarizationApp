import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.chains.summarize import load_summarize_chain
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline
import torch
import base64

# chekcpoint = "google/flan-t5-base"
checkpoint = "LaMini-Flan-T5-248M"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
# model = T5ForConditionalGeneration.from_pretrained(chekcpoint)
# summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, device_map="auto", torch_dtype=torch.float32)

#file loader and preprocessing
def file_preprocessing(file):
    loader =  PyPDFLoader(file)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    texts = text_splitter.split_documents(pages)
    final_texts = ""
    for text in texts:
        print(text)
        final_texts = final_texts + text.page_content
    return final_texts

#LLM pipeline
def llm_pipeline(filepath):
    pipe_sum = pipeline(
        'summarization',
        model = base_model,
        tokenizer = tokenizer,
        max_length = 500, 
        min_length = 50)
    input_text = file_preprocessing(filepath)
    result = pipe_sum(input_text)
    result = result[0]['summary_text']
    return result

@st.cache_data
#function to display the PDF of a given file 
def displayPDF(file):
    # Opening file from file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    # Embedding PDF in HTML
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)
    
#streamlit code 
st.set_page_config(layout="wide", page_title="Summarization App")

def run():
    st.title("Text Summarization App")
    
    st.sidebar.title("Options")
    option = st.sidebar.selectbox(
        "Select an option", 
        ("Text", "Document"), 
        index=None,
    )
    
    if option == "Document":
        uploaded_file = st.file_uploader("Upload your PDF file", type=['pdf'])
        
        if uploaded_file is not None:
            if st.button("Summarize"):
                col1, col2 = st.columns(2)
                filepath = "data/"+uploaded_file.name
                with open(filepath, "wb") as temp_file:
                    temp_file.write(uploaded_file.read())
                with col1:
                    st.info("Uploaded File")
                    pdf_view = displayPDF(filepath)

                with col2:
                    st.info("Summarization Complete")
                    
                    summary = llm_pipeline(filepath)
                    st.success(summary)
    elif option == "Text":
        text = st.text_area("Enter Text Content:", height=200)
        
        if st.button("Summarize"):
            if text.strip():
                # Thêm prefix "summarize: " theo yêu cầu của model
                input_text = "summarize: " + text

                # Tokenize đầu vào
                inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

                # Tạo bản tóm tắt
                summary_ids = base_model.generate(**inputs, max_length=150, min_length=30, length_penalty=2.0, num_beams=4)

                # Giải mã kết quả
                summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

                # Hiển thị kết quả
                st.subheader("Summary:")
                st.write(summary)
            else:
                st.warning("Please enter text content.")

    



# if __name__ == "__main__":
#     main()