import streamlit as st
import os
import faiss
from io import BytesIO
from docx import Document
import numpy as np
import google.generativeai as genai
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

# Initialize Gemini with safety settings (moved outside of answer_question)
genai.configure(api_key=GOOGLE_API_KEY)
generation_config = genai.types.GenerationConfig(
    temperature=0.5,
    top_p=0.8,
    top_k=40,
    max_output_tokens=1024,
)
safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}

gemini_model = genai.GenerativeModel(
    model_name="gemini-pro",
    generation_config=generation_config,
    safety_settings=safety_settings,
)

# Save the model to the session state
st.session_state['gemini_model'] = gemini_model
st.session_state['embedding_model'] = genai.GenerativeModel("models/embedding-001")

# Helper functions
def load_file(file, file_type):
    """Load content from a file (PDF, DOCX, or TXT)."""
    try:
        if file_type == "PDF":
            pdf_reader = PdfReader(BytesIO(file.read()))
            return "".join([page.extract_text() for page in pdf_reader.pages])
        elif file_type == "DOCX":
            doc = Document(BytesIO(file.read()))
            return "\n".join([para.text for para in doc.paragraphs])
        elif file_type == "TXT":
            return file.read().decode("utf-8")
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    except Exception as e:
        raise ValueError(f"Error processing {file_type} file: {e}")

def process_input(input_type, input_data):
    """Process the input data based on the input type and create a vector store."""
    try:
        start_time = time.time()
        logging.info(f"Starting input processing for {input_type}")
        # Handle input types
        if input_type == "Web":
            loader = WebBaseLoader(input_data)
            documents = loader.load()
        else:
            documents = load_file(input_data, input_type)
        end_time = time.time()
        logging.info(f"Loaded Documents in {end_time - start_time:.2f} seconds")

        # Split documents into chunks using RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        if input_type == "Web":
            texts = [str(doc.page_content) for doc in text_splitter.split_documents(documents)]
        else:
            texts = text_splitter.split_text(documents)
        
        end_time = time.time()
        logging.info(f"Split texts in {end_time - start_time:.2f} seconds")
        
        # Get Gemini embeddings
        embedding_model = st.session_state['embedding_model']

        def get_gemini_embeddings(texts):
          results = embedding_model.embed_content(texts)
          return np.array([result['embedding'] for result in results['embeddings']])

        # Create FAISS index
        sample_embedding = get_gemini_embeddings(["sample text"])
        dimension = sample_embedding.shape[1]
        index = faiss.IndexFlatL2(dimension)

        # Create FAISS vector store
        vector_store = FAISS(
            embedding_function=get_gemini_embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        
        # Add documents to the vector store in batches
        batch_size = 20
        for i in range(0, len(texts), batch_size):
            vector_store.add_texts(texts[i:i + batch_size])
        
        end_time = time.time()
        logging.info(f"Created Vector Store in {end_time - start_time:.2f} seconds")

        return vector_store

    except Exception as e:
        raise RuntimeError(f"Error during input processing: {e}")
    finally:
        logging.info(f"Finished processing input for {input_type}")

def answer_question(vectorstore, query):
    """Answers a question based on the provided vector store."""
    try:
        llm = lambda x: st.session_state['gemini_model'].generate_content(x).text
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
        start_time = time.time()
        result = qa({"query": query})
        end_time = time.time()
        logging.info(f"Generated answer in {end_time - start_time:.2f} seconds")
        return result
    except Exception as e:
        raise RuntimeError(f"Error generating answer: {e}")

def save_vectorstore(vectorstore, filepath):
    """Saves the FAISS vector store to a file."""
    try:
        with open(filepath, "wb") as f:
          pickle.dump(vectorstore, f)
        logging.info(f"Vector store saved to {filepath}")
    except Exception as e:
        raise RuntimeError(f"Error saving vectorstore: {e}")

def load_vectorstore(filepath):
    """Loads the FAISS vector store from a file."""
    try:
      if not os.path.exists(filepath):
          logging.info(f"Vector store not found at {filepath}")
          return None
      with open(filepath, "rb") as f:
            vectorstore = pickle.load(f)
      logging.info(f"Vector store loaded from {filepath}")
      return vectorstore
    except Exception as e:
      raise RuntimeError(f"Error loading vectorstore: {e}")

# Main application
def main():
    st.title("40-Tech RAG Q&A App (Gemini) 🚀")
    
    # Select input type
    input_type = st.selectbox("Select a source", ["Web", "PDF", "DOCX", "Text", "TXT"])
    input_data = None

    if input_type == "Web":
        number_input = st.number_input("Number of URLs", min_value=1, max_value=10, step=1)
        input_data = [st.text_input(f"Enter URL {i+1}") for i in range(number_input)]


    # Filepath for the FAISS index, you can make this a user input for persistence
    vectorstore_filepath = "faiss_index.pkl"
    vectorstore = None
    if 'vectorstore' in st.session_state:
      vectorstore = st.session_state['vectorstore']
    else:
      vectorstore = load_vectorstore(vectorstore_filepath)
      if vectorstore:
        st.session_state['vectorstore'] = vectorstore
        st.success("Vector store loaded successfully!")

    if st.button("Process"):
      if not input_data:
            st.error("Please provide valid input.")
            return

      with st.spinner("Processing input..."):
          try:
                vectorstore = process_input(input_type, input_data)
                st.session_state["vectorstore"] = vectorstore
                save_vectorstore(vectorstore, vectorstore_filepath)
                st.success("Vector store created and saved successfully!")
          except Exception as e:
                st.error(f"Error processing input: {e}")
                return

    # Question-answering section
    if "vectorstore" in st.session_state:
      query = st.text_input("Ask your question")
      if st.button("Submit"):
            with st.spinner("Generating answer..."):
                try:
                    answer = answer_question(st.session_state["vectorstore"], query)
                    st.write(f"**Answer**: {answer}")
                except Exception as e:
                    st.error(f"Error generating answer: {e}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")