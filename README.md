# 🧠 MediBot — End-to-End Medical Chatbot using RAG

MediBot is an **end-to-end Retrieval-Augmented Generation (RAG)** based chatbot built using **LangChain**, **FAISS**, and **Mistral AI**.  
It retrieves relevant medical context from a local knowledge base and generates precise, conversational answers — powered by modern open-source LLMs.

---

## 🚀 Features

- 🔍 **Retrieval-Augmented Generation (RAG)** — Combines context retrieval + LLM reasoning.
- 🧩 **FAISS Vector Store** — Fast and local document similarity search.
- 🤖 **Mistral LLM Integration** — Uses `mistral-large-latest` for high-quality, low-latency responses.
- ⚙️ **LangChain Runnable Pipeline** — Uses the new composable API (`retriever → prompt → llm`).
- 💬 **Flask Web App** — Simple, lightweight chat interface.
- 🔐 **Environment Variable Integration** — For secure API key handling.
- 🧠 **HuggingFace Embeddings** — Converts text into dense vectors using transformer-based encoders.

---

## 🧩 Architecture

**Pipeline Flow:**
1. Extract and split documents into chunks.
2. Generate embeddings using HuggingFace models.
3. Store embeddings in a FAISS index.
4. On user query:
   - Retrieve top-k relevant chunks.
   - Inject them into a prompt template.
   - Pass to Mistral for context-aware answer generation.

---

## 🛠️ Tech Stack

| Component | Tool / Library |
|------------|----------------|
| Framework | Flask |
| LLM | Mistral (via LangChain integration) |
| Embeddings | HuggingFace Transformers |
| Vector Store | FAISS |
| Orchestration | LangChain Runnable API |
| Environment Management | Python-dotenv |

---

## ⚙️ Setup Instructions

### 1️⃣ Clone Repository
```bash
git clone https://github.com/<your-username>/MediBot.git
cd MediBot
```

---

### 2️⃣ Create Virtual Environment
```bash
conda create -n medibot python=3.10 -y
conda activate medibot
```

---

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

---

### 4️⃣ Setup Environment Variables
Create a `.env` file in the project root and add your keys:
```bash
MISTRAL_API_KEY=your_mistral_api_key
```

---

### 5️⃣ (Optional) Add More Data & Update FAISS Index
If you want to expand MediBot’s knowledge base:  
1. Add your documents to the **`Data/`** folder.  
2. Run the embedding script to generate vectors and update FAISS:

```
python store_embedding.py
---
This will update the faiss_index/ directory.
Once done, app.py will automatically use the updated embeddings for retrieval-augmented responses.

### 6️⃣ Launch Flask App
```bash
python app.py
```

The app will be available at:
👉 **http://127.0.0.1:5000**

---

## 📦 Project Structure
MediBot/
│
├── app.py                     # Flask backend (main entry)
├── src/
│   ├── helper.py              # Embedding loader + FAISS helpers
│   ├── prompt.py              # Prompt template (system + human)
│
├── faiss_index/               # Stored FAISS vector database
├── templates/
│   └── chat.html              # Frontend UI
├── requirements.txt
└── .env