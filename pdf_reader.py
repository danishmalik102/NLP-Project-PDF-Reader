from dotenv import load_dotenv
import os
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Load environment variables from .env file
load_dotenv()

# Sidebar contents
with st.sidebar:
    st.title('💬  Chat With Your PDF')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
    ''')
    add_vertical_space(5)
    st.write('Made By Danish Malik and Muhammad Sameer')
    

def main():
    st.header("Chat with PDF")

    # Upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        st.write(text)

        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)
        st.write(chunks)

        # Embeddings
        embeddings = OpenAIEmbeddings()
        KnowledgeBase = FAISS.from_texts(chunks, embedding=embeddings)

        # Show user input
        user_question = st.text_input("Ask a question from your PDF:")
        if user_question:
            docs = KnowledgeBase.similarity_search(user_question)
            st.write(docs)

if __name__ == '__main__':
    main()
