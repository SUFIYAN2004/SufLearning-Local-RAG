#  SufLearning: Local RAG Pipeline

A fully offline, private Retrieval-Augmented Generation (RAG) document companion that runs entirely on your local Windows PC. **No internet connection, no cloud APIs, no login screens, and no API keys required.**[cite: 1] You upload a PDF, the application indexes it into a local vector database, and a local language model answers your questions using only the content of that document[cite: 1].

> **Privacy First:** Because the document text, embedding vectors, and model inference stay entirely on your local machine, this pipeline ensures absolute data privacy.

---

## 🗺️ Architecture Blueprint

The pipeline is split into a one-time ingestion path (top row) and a runtime chat loop (bottom row)[cite: 1]:

```text
📁 PDF UPLOAD (Streamlit) ──> 📄 TEXT EXTRACT (pypdf) ──> ✂️ CHUNKING (LangChain) ──> 🧮 EMBEDDING ──> 📦 VECTOR DB (ChromaDB)
                                                                                                        │
📥 USER QUERY ────────────────> 🔍 RETRIEVAL (Top-10 Chunks) ──> 📝 PROMPT BUILD 💾 <──────────────────┘
                                                                        │
💬 CHAT DISPLAY <────────────── ⚡ STREAM REPLY <────────────── 🔥 llama-server.exe (Gemma 3 4B GGUF)

```

Every component runs locally on `localhost`.

---

## ⚡ Tech Stack & Building Blocks

* **Frontend/UI:** [Streamlit](https://streamlit.io/) — Interactive web interface with chat bubbles and file upload handling.


* **Text Processing:** `pypdf` + LangChain `RecursiveCharacterTextSplitter` — For extracting text and splitting it into 1500-character chunks with a 300-character overlap.


* **Vector Storage:** [ChromaDB](https://www.trychroma.com/) — Persistent on-disk vector database storing text embeddings completely offline.


* **Local Inference Engine:** `llama.cpp` (Vulkan Build) — Cross-platform LLM server providing hardware acceleration for non-NVIDIA/AMD/Intel GPUs.


* **The Brain:** Google Gemma 3 4B IT (GGUF Format) — Quantized down to 3-bit (`Q3_K_M`) for high-speed, lightweight performance.



---

## 🛠️ System Prerequisites

* **OS:** Windows 10/11 (64-bit)


* **Hardware:** A modern GPU supporting **Vulkan** (NVIDIA, AMD, or Intel) to offload model layers.


* **Software:** Python 3.10 or newer (ensure it is added to your system environment variables).


* **Storage:** ~5 GB of free space for the server binaries and the GGUF model weight file.



---

## 🚀 Quick Start Guide

### Step 1: Set Up the Inference Server

1. Download the Vulkan Windows binary version of `llama.cpp` from the official [GitHub Releases](https://github.com/ggml-org/llama.cpp/releases) page.


2. Extract the archive to a simple directory, such as `C:\llama\`.


3. Download the `gemma-3-4b-it-Q3_K_M.gguf` file from a community mirror on Hugging Face (e.g., bartowski or unsloth) and store it safely.


4. Open a terminal inside your `llama.cpp` directory and run the following command to spin up the local OpenAI-compatible API server:



```powershell
.\llama-server.exe --model "C:\path\to\your\models\gemma-3-4b-it-Q3_K_M.gguf" --ctx-size 8192 -ngl 99 -t 4

```

*Leave this terminal window running in the background.*

### Step 2: Clone & Install Python Environment

Open a second terminal window, navigate to your project directory, and configure the workspace:

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install the dependencies
pip install -r requirements.txt

```

### Step 3: Run the Interface

Launch the application wrapper using Streamlit:

```bash
streamlit run app.py

```

Your browser will open up to `http://localhost:8501` automatically. Drop a document into the sidebar, index it, and start chatting!

---

## 🛠️ Troubleshooting

| Symptom | Common Root Cause | Fix Action |
| --- | --- | --- |
| `Failed to connect to local llama-server` | Server is either closed or binded to another port.| Double-check that your server terminal is running on port 8080.|
| Extremely slow generation | Execution fell back onto CPU threads.| Confirm Vulkan drivers are active and `-ngl 99` flag is present.|
| `No context found` Warning | Vector store is empty.| Ensure you clicked the **"Process & Index PDF"** button in the sidebar after upload.|

---

*Compiled & Developed by V. Mohammed Sufiyan*
