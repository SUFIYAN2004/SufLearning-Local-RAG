import streamlit as st [cite: 109, 226]
import os [cite: 110, 227]
from pypdf import PdfReader [cite: 111, 228]
from langchain_text_splitters import RecursiveCharacterTextSplitter [cite: 112, 229]
import chromadb [cite: 112, 230]
from chromadb.utils import embedding_functions [cite: 113, 231]
from openai import OpenAI [cite: 114, 232]

# Configuration Constants
CHROMA_DATA_PATH = "./chroma_db" [cite: 115, 233]
COLLECTION_NAME = "Team_Sufiyan" [cite: 116, 234]
LLAMA_SERVER_URL = "http://localhost:8080/v1" [cite: 117, 235]

# Initialize OpenAI Client pointing to the local llama-server
client = OpenAI(base_url=LLAMA_SERVER_URL, api_key="not-needed") [cite: 118, 236]

# Initialize Local ChromaDB Persistent Client
chroma_client = chromadb.PersistentClient(path=CHROMA_DATA_PATH) [cite: 138, 237]
default_ef = embedding_functions.DefaultEmbeddingFunction() [cite: 139, 140, 238, 239]
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=default_ef
) [cite: 141, 142, 143, 144, 240, 241, 242]

# Page Configuration & UI Styling
st.set_page_config(page_title="SufLearning Local RAG", page_icon="🤖", layout="wide") [cite: 145, 147, 244]

st.markdown("""
<style>
main { background-color: #0e1117; color: #ffffff; }
.stTextInput>div>div>input { background-color: #1a1c23; color: #ffffff; border-color: #4f46e5; }
.stButton>button { background-color: #4f46e5; color: white; border-radius: 8px; }
</style>
""", unsafe_allow_html=True) [cite: 245, 246, 247, 248, 249, 250, 251]

st.title("🤖 SufLearning: Local RAG Pipeline") [cite: 252]
st.caption("Gemma 3 4B (Vulkan) + ChromaDB + Streamlit | 100% Offline & Private") [cite: 253]

# Sidebar for PDF Document Ingestion
with st.sidebar: [cite: 154, 254]
    st.header("📂 Document Ingestion") [cite: 255]
    uploaded_file = st.file_uploader("Upload a PDF to your local vector store", type=["pdf"]) [cite: 155, 256]
    
    if uploaded_file is not None: [cite: 155, 257]
        if st.button("Process & Index PDF"): [cite: 156, 258]
            with st.spinner("Processing document completely offline..."): [cite: 259]
                
                # Extract text from PDF
                pdf_reader = PdfReader(uploaded_file) [cite: 157, 260]
                raw_text = "" [cite: 158, 261]
                for page in pdf_reader.pages: [cite: 159, 262]
                    text = page.extract_text() [cite: 160, 263]
                    if text: [cite: 161, 264]
                        raw_text += text [cite: 162, 265]
                
                # Split text into manageable overlapping chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1500,
                    chunk_overlap=300,
                    length_function=len
                ) [cite: 163, 165, 267, 269, 270, 271]
                chunks = text_splitter.split_text(raw_text) [cite: 166, 272]
                
                # Refresh collection to avoid mixing old and new documents
                try: [cite: 273]
                    chroma_client.delete_collection(COLLECTION_NAME) [cite: 167, 274]
                    collection = chroma_client.get_or_create_collection(
                        name=COLLECTION_NAME,
                        embedding_function=default_ef
                    ) [cite: 168, 275, 276]
                except Exception: [cite: 277]
                    pass [cite: 278]
                
                # Prepare metadata and structural payloads for vector storage
                documents_list = chunks [cite: 279]
                metadatas_list = [{"source": uploaded_file.name} for _ in chunks] [cite: 280]
                ids_list = [f"chunk_{i}" for i in range(len(chunks))] [cite: 281]
                
                # Commit text and mathematical vectors into ChromaDB
                collection.add(
                    documents=documents_list,
                    metadatas=metadatas_list,
                    ids=ids_list
                ) [cite: 169, 282, 284, 285, 286]
                
                st.success(f"Indexed {len(chunks)} text chunks into ChromaDB successfully!") [cite: 287]

# Main Chat Workspace Area
st.subheader("💬 Chat with your Knowledge Base") [cite: 288]

# Manage application runtime session chat history
if "messages" not in st.session_state: [cite: 180, 289]
    st.session_state.messages = [] [cite: 180, 290]

# Render persistent historical context on display update
for message in st.session_state.messages: [cite: 181, 291]
    with st.chat_message(message["role"]): [cite: 182, 292]
        st.markdown(message["content"]) [cite: 182, 293]

# Core Query Loop
if user_query := st.chat_input("Ask something about your documents..."): [cite: 183, 294]
    with st.chat_message("user"): [cite: 295]
        st.markdown(user_query) [cite: 296]
    st.session_state.messages.append({"role": "user", "content": user_query}) [cite: 183, 297]
    
    # Retrieve contextual references matching the query embedding space
    try: [cite: 298]
        results = collection.query(
            query_texts=[user_query],
            n_results=10
        ) [cite: 183, 299, 300, 302, 303]
        context_chunks = results['documents'][0] if results['documents'] else [] [cite: 184, 304]
        context_str = "\n--\n".join(context_chunks) [cite: 185, 304]
    except Exception: [cite: 305]
        context_str = "" [cite: 306]
        st.warning("No context found. Database might be empty. Processing query without context.") [cite: 307]
    
    # System Instruction Design Blueprint
    system_prompt = (
        "You are a helpful, precise local AI assistant. Answer the user's question using only the provided context.\n"
        "If the answer cannot be found in the context, explicitly state that you don't know based on the provided data.\n"
        "Do not make up facts."
    ) [cite: 199, 308, 309, 310, 311, 312]
    
    full_prompt = f"Context:\n{context_str}\n\nQuestion: {user_query}\nAnswer:" [cite: 200, 313]
    
    # Run Streaming Answer Synthesis via Local LLM API endpoint
    with st.chat_message("assistant"): [cite: 314]
        response_placeholder = st.empty() [cite: 315]
        full_response = "" [cite: 316]
        
        try: [cite: 317]
            stream = client.chat.completions.create(
                model="gemma-3-4b", [cite: 202, 319]
                messages=[
                    {"role": "system", "content": system_prompt}, [cite: 204, 321]
                    {"role": "user", "content": full_prompt} [cite: 205, 321]
                ], [cite: 203, 322]
                stream=True, [cite: 206, 324]
                max_tokens=1024, [cite: 206, 325]
                temperature=0.3 [cite: 206, 326]
            ) [cite: 201]
            
            # Catch token deltas as they break off the server line stream
            for chunk in stream: [cite: 208, 327]
                if chunk.choices[0].delta.content: [cite: 209, 328]
                    full_response += chunk.choices[0].delta.content [cite: 210, 329]
                    response_placeholder.markdown(full_response + "▌") # Cursor simulation
            response_placeholder.markdown(full_response) [cite: 330]
            
        except Exception as e: [cite: 331]
            st.error(f"Failed to connect to local llama-server. Ensure it's running on port 8080. Error: {e}") [cite: 332, 333]
        
        # Save valid model answers back into session historical dictionary
        if full_response: [cite: 334]
            st.session_state.messages.append({"role": "assistant", "content": full_response}) [cite: 335]
