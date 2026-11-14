import os
import tempfile
import streamlit as st

from dotenv import load_dotenv
from pypdf import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="PDF RAG Chatbot")
st.title("📄 PDF RAG Chatbot with Gemini")

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type="pdf"
)

if uploaded_file:

    pdf_reader = PdfReader(uploaded_file)

    text = ""

    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    st.success("PDF Loaded Successfully!")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_text(text)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )

    vector_store = FAISS.from_texts(
        chunks,
        embedding=embeddings
    )

    question = st.text_input(
        "Ask a question about the PDF"
    )

    if question:

        docs = vector_store.similarity_search(
            question,
            k=4
        )

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0.3
        )

        chain = load_qa_chain(
            llm,
            chain_type="stuff"
        )

        response = chain.run(
            input_documents=docs,
            question=question
        )

        st.subheader("Answer")
        st.write(response)