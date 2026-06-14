# 🧠 LangGraph RAG Chatbot

A RAG (Retrieval-Augmented Generation) chatbot built with **LangChain + LangGraph**, featuring a Streamlit UI and FAISS vector store.

## 📁 Project Structure

```
langgraph_rag/
├── app.py              # Main Streamlit app + LangGraph pipeline
├── requirements.txt    # Dependencies
├── .env                # Your OpenAI API key (never commit this)
├── .env.example        # Template for the .env file
├── .gitignore          # Excludes .env, venv, vectorstore, pycache
├── data/
│   └── knowledge.txt/or any other file   # Default knowledge base (add your own .txt or .pdf files)
└── README.md
```

## 🗺️ LangGraph Flow

```
[retrieve]          → Fetch top-4 relevant chunks from FAISS
    ↓
[grade]             → Check if chunks are relevant to the question
    ↓ conditional
  ┌─────────────────────────┐
  ↓                         ↓
[build_context]          [fallback]
  ↓                         ↓
[generate]               LLM answers from general knowledge
  ↓
 END
```

## 🚀 Setup & Run

### 1. Clone the repo
```bash
git clone https://github.com/ftahir85/langgraph-rag.git
cd langgraph-rag
```

### 2. Create and activate virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your OpenAI API key
Create a `.env` file in the project root:
```bash
OPENAI_API_KEY=sk-your-key-here
```

### 5. Add documents to the knowledge base
Drop `.txt` or `.pdf` files into the `data/` folder.

### 6. Run the app
```bash
streamlit run app.py
```

## 💡 How to Add More Documents

**Option A — Via the sidebar UI:**
- Upload a `.txt` or `.pdf` file using the sidebar uploader
- The app automatically clears the old index and rebuilds with the new file

**Option B — Manually:**
- Drop files directly into the `data/` folder
- Delete the `vectorstore/` folder if it exists
- Restart the app — the index rebuilds automatically

## ⚡ FAISS Disk Caching

On first run, the app embeds all documents and saves the FAISS index to disk:
```
vectorstore/
└── faiss_index/
    ├── index.faiss
    └── index.pkl
```
Every restart after that loads the index from disk instantly — no re-embedding, no extra API calls.


- [OpenAI GPT-4o-mini](https://platform.openai.com/docs/models) — answer generation
- [Streamlit](https://streamlit.io/) — UI
