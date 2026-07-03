import streamlit as st 
import os 
from pypdf import PdfReader 
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions 
from openai import OpenAI 

# Configuration Constants
CHROMA_DATA_PATH = "./chroma_db" 
COLLECTION_NAME = "Team_Sufiyan" 
LLAMA_SERVER_URL = "http://localhost:8080/v1"

# Initialize OpenAI Client pointing to the local llama-server
client = OpenAI(base_url=LLAMA_SERVER_URL, api_key="not-needed") 

# Initialize Local ChromaDB Persistent Client
chroma_client = chromadb.PersistentClient(path=CHROMA_DATA_PATH) 
default_ef = embedding_functions.DefaultEmbeddingFunction() 
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=default_ef
) 

# Page Configuration & UI Styling
st.set_page_config(page_title="SufLearning Local RAG", page_icon="🤖", layout="wide") 

st.markdown("""
<style>
main { background-color: #0e1117; color: #ffffff; }
.stTextInput>div>div>input { background-color: #1a1c23; color: #ffffff; border-color: #4f46e5; }
.stButton>button { background-color: #4f46e5; color: white; border-radius: 8px; }
</style>
""", unsafe_allow_html=True) 

st.title("🤖 SufLearning: Local RAG Pipeline") 
st.caption("Gemma 3 4B (Vulkan) + ChromaDB + Streamlit | 100% Offline & Private") 

# Sidebar for PDF Document Ingestion
with st.sidebar: 
    st.header("📂 Document Ingestion")
    uploaded_file = st.file_uploader("Upload a PDF to your local vector store", type=["pdf"])
    
    if uploaded_file is not None: 
        if st.button("Process & Index PDF"): 
            with st.spinner("Processing document completely offline..."): 
                
                # Extract text from PDF
                pdf_reader = PdfReader(uploaded_file) 
                raw_text = "" 
                for page in pdf_reader.pages: 
                    text = page.extract_text() 
                    if text: 
                        raw_text += text 
                
                # Split text into manageable overlapping chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1500,
                    chunk_overlap=300,
                    length_function=len
                ) 
                chunks = text_splitter.split_text(raw_text)
                
                # Refresh collection to avoid mixing old and new documents
                try: 
                    chroma_client.delete_collection(COLLECTION_NAME) 
                    collection = chroma_client.get_or_create_collection(
                        name=COLLECTION_NAME,
                        embedding_function=default_ef
                    ) 
                except Exception: 
                    pass 
                # Prepare metadata and structural payloads for vector storage
                documents_list = chunks
                metadatas_list = [{"source": uploaded_file.name} for _ in chunks]
                ids_list = [f"chunk_{i}" for i in range(len(chunks))] 
                
                # Commit text and mathematical vectors into ChromaDB
                collection.add(
                    documents=documents_list,
                    metadatas=metadatas_list,
                    ids=ids_list
                ) 
                
                st.success(f"Indexed {len(chunks)} text chunks into ChromaDB successfully!") 

# Main Chat Workspace Area
st.subheader("💬 Chat with your Knowledge Base") 

# Manage application runtime session chat history
if "messages" not in st.session_state: 
    st.session_state.messages = [] 

# Render persistent historical context on display update
for message in st.session_state.messages: 
    with st.chat_message(message["role"]):
        st.markdown(message["content"]) 

# Core Query Loop
if user_query := st.chat_input("Ask something about your documents..."):
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Retrieve contextual references matching the query embedding space
    try:
        results = collection.query(
            query_texts=[user_query],
            n_results=10
        ) 
        context_chunks = results['documents'][0] if results['documents'] else []
        context_str = "\n--\n".join(context_chunks) 
    except Exception: 
        context_str = "" 
        st.warning("No context found. Database might be empty. Processing query without context.") 
    
    # System Instruction Design Blueprint
    system_prompt = (
        "You are a helpful, precise local AI assistant. Answer the user's question using only the provided context.\n"
        "If the answer cannot be found in the context, explicitly state that you don't know based on the provided data.\n"
        "Do not make up facts."
    ) 
    
    full_prompt = f"Context:\n{context_str}\n\nQuestion: {user_query}\nAnswer:"
    
    # Run Streaming Answer Synthesis via Local LLM API endpoint
    with st.chat_message("assistant"): 
        response_placeholder = st.empty() 
        full_response = "" 
        
        try:
            stream = client.chat.completions.create(
                model="gemma-3-4b", 
                messages=[
                    {"role": "system", "content": system_prompt}, 
                    {"role": "user", "content": full_prompt} 
                ], 
                stream=True, 
                max_tokens=1024,
                temperature=0.3
            ) 
            
            # Catch token deltas as they break off the server line stream
            for chunk in stream: 
                if chunk.choices[0].delta.content: 
                    full_response += chunk.choices[0].delta.content 
                    response_placeholder.markdown(full_response + "▌") # Cursor simulation
            response_placeholder.markdown(full_response) 
            
        except Exception as e: 
            st.error(f"Failed to connect to local llama-server. Ensure it's running on port 8080. Error: {e}")
        
        # Save valid model answers back into session historical dictionary
        if full_response: 
            st.session_state.messages.append({"role": "assistant", "content": full_response})
