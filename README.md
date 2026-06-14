# 🧠 LangGraph RAG Chatbot

A RAG (Retrieval-Augmented Generation) chatbot built with **LangChain + LangGraph**, featuring a Streamlit UI and FAISS vector store.

## 📁 Project Structure

```
langgraph_rag/
├── app.py              # Main Streamlit app + LangGraph pipeline
├── requirements.txt    # Dependencies
├── data/
│   └── knowledge.txt   # Default knowledge base (add your own .txt or .pdf files)
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

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
streamlit run app.py
```

### 3. Enter your OpenAI API key in the sidebar

## 💡 How to Add More Documents

- Drop `.txt` or `.pdf` files into the `data/` folder
- Restart the app — the vector store rebuilds automatically

## 🔍 LangChain vs Your From-Scratch RAG

| Feature              | Your RAG Project        | This Project               |
|----------------------|-------------------------|----------------------------|
| Embeddings           | sentence-transformers   | OpenAI text-embedding       |
| Vector Store         | FAISS (manual)          | FAISS via LangChain         |
| Chunking             | Manual                  | RecursiveCharacterTextSplitter |
| LLM Call             | openai SDK directly     | ChatOpenAI (LangChain)      |
| Flow Control         | Linear Python           | LangGraph StateGraph        |
| Conditional Routing  | if/else                 | Conditional edges in graph  |
