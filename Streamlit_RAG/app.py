import streamlit as st
import os
import faiss
from io import BytesIO
from docx import Document
import numpy as np
import google.generativeai as genai
from google.generativeai.generative_models import GenerativeModel
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from langchain_community.document_loaders import WebBaseLoader
from PyPDF2 import PdfReader
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from dotenv import load_dotenv
import time
import logging
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Streamlit page config
st.set_page_config(page_title="40-Tech RAG Q&A App (Gemini)", page_icon="🤖")

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Validate the Gemini API key
if not GOOGLE_API_KEY:
    st.error("Google Gemini API key not found. Please set it in the environment variables.")
    st.stop()
    
# Initialize Gemini &Create the model
generation_config = {
  "temperature": .5,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-2.0-flash-exp",
  generation_config=generation_config,
  system_instruction="You are a helpful assistant the responds to questions about uploaded documents.",
)

# Save the model to the session state
st.session_state['gemini_model'] = model
st.session_state['embedding_model'] = genai.GenerativeModel("models/embedding-001")



    
def main():
    st.title("40-Tech RAG Q&A App (Gemini) 🚀")
    
    # Select input type
    input_type = st.selectbox("Select a source", ["Web", "PDF", "DOCX", "Text", "TXT"])
    input_data = None
    
    if input_type == "Web":
        number_input = st.number_input("Number of pages to scrape (up to 5)", min_value=1, max_value=5, value=1)
        input_data = [st.sidebar.text_input(f"Enter URL {i+1}") for i in range(number_input)]
    elif input_type in ["PDF", "DOCX", "TXT"]:
        input_data = st.file_uploader(f"Upload a {input_type} file", type=input_type.lower())
    elif input_type == "Text":
        input_data = st.text_area("Enter the text")
    else:
        st.error("Invalid input type")
        return
    
    if st.button("Process"):
        if not input_data:
            st.error("Please provide valid input")
            return
    
if __name__ == "__main__":
    main()