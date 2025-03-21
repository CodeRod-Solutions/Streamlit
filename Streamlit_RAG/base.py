import streamlit as st
import google.generativeai as genai
import tempfile
import io
import requests
import PyPDF2
import docx2txt
from typing import List, Dict, Any
import os

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    raise EnvironmentError("GEMINI_API_KEY environment variable not set.")

genai.configure(api_key=GEMINI_API_KEY)


# Retrieve canonical list of Gemini models
available_models = genai.list_models()

# Display in UI select box
selected_model = st.selectbox("Select a Gemini Model", available_models)

st.write("You selected:", selected_model)

# Dummy function to simulate Google Gemini embedding.
def gemini_embed(text: str) -> List[float]:
    # In a real system, you would call the Google Gemini API here
    # to transform the text into an embedding vector.
    # For now, we simply return a placeholder vector.
    return [float(len(text))]  # Dummy vector based on text length

class InMemoryVectorDB:
    def __init__(self):
        # Each record: {'embedding': vector, 'text': original_text, 'source': source_identifier}
        self.records: List[Dict[str, Any]] = []

    def add_document(self, text: str, source: str):
        embedding = gemini_embed(text)
        self.records.append({
            'embedding': embedding,
            'text': text,
            'source': source
        })

    def similarity_search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        query_embedding = gemini_embed(query)
        # Dummy similarity: difference between embedding values.
        # Replace this with a proper cosine similarity in real use.
        def score(record):
            return abs(record['embedding'][0] - query_embedding[0])
        ranked = sorted(self.records, key=score)
        return ranked[:top_k]

# Helper functions to extract text from various sources.
def extract_text_from_pdf(file_stream) -> str:
    reader = PyPDF2.PdfReader(file_stream)
    text = ""
    for page in reader.pages:
        text += page.extract_text() if page.extract_text() else ""
    return text

def extract_text_from_doc(file_stream) -> str:
    # For .docx files, using docx2txt
    file_bytes = file_stream.read()
    return docx2txt.process(io.BytesIO(file_bytes))

def extract_text_from_txt(file_stream) -> str:
    return file_stream.read().decode("utf-8")

def fetch_text_from_url(url: str) -> str:
    response = requests.get(url)
    response.raise_for_status()
    return response.text

def main():
    st.title("RAG System with Google Gemini & In-Memory Vector DB")
    
    vector_db = InMemoryVectorDB()
    
    st.header("Upload Documents or Enter URL / Free Text")
    option = st.radio("Choose input type:", ["Upload File", "URL", "Free Text"])
    
    if option == "Upload File":
        uploaded_file = st.file_uploader("Choose a file (.pdf, .txt, .docx)", type=["pdf", "txt", "docx"])
        if uploaded_file is not None:
            file_type = uploaded_file.type
            try:
                if "pdf" in file_type:
                    text = extract_text_from_pdf(uploaded_file)
                elif "plain" in file_type:
                    text = extract_text_from_txt(uploaded_file)
                elif "word" in file_type or "doc" in file_type:
                    text = extract_text_from_doc(uploaded_file)
                else:
                    st.error("Unsupported file type.")
                    return
                vector_db.add_document(text, uploaded_file.name)
                st.success(f"Document '{uploaded_file.name}' added to the vector database.")
            except Exception as e:
                st.error(f"Error processing file: {e}")
    
    elif option == "URL":
        url = st.text_input("Enter URL:")
        if url and st.button("Fetch and Add"):
            try:
                text = fetch_text_from_url(url)
                vector_db.add_document(text, url)
                st.success("Content from URL added to the vector database.")
            except Exception as e:
                st.error(f"Error fetching URL: {e}")
    
    elif option == "Free Text":
        free_text = st.text_area("Enter free text:")
        if free_text and st.button("Add Text"):
            vector_db.add_document(free_text, "free_text_input")
            st.success("Free text added to the vector database.")
    
    st.header("Query the RAG System")
    query = st.text_input("Enter your query:")
    if st.button("Search") and query:
        results = vector_db.similarity_search(query)
        if results:
            st.subheader("Retrieved Documents:")
            for idx, record in enumerate(results):
                st.markdown(f"**Result {idx+1}:** Source: {record['source']}")
                st.write(record['text'][:500] + "...")
        else:
            st.info("No matching documents found.")

if __name__ == '__main__':
    main()