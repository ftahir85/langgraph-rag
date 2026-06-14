import os
import shutil
import streamlit as st
from typing import TypedDict, List
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# LangChain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.documents import Document

# LangGraph imports
from langgraph.graph import StateGraph, END

# ─────────────────────────────────────────────
# 1. LANGGRAPH STATE
# ─────────────────────────────────────────────
class RAGState(TypedDict):
    question: str
    retrieved_docs: List[Document]
    context: str
    answer: str
    has_relevant_docs: bool


# ─────────────────────────────────────────────
# 2. VECTOR STORE — FAISS disk saving
# ─────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
FAISS_PATH  = os.path.join(BASE_DIR, "vectorstore", "faiss_index")
DATA_DIR    = os.path.join(BASE_DIR, "data")

@st.cache_resource(show_spinner="⚡ Loading vector store...")
def build_vectorstore():
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

    if os.path.exists(FAISS_PATH):
        return FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True), "loaded"

    docs = []
    for fname in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, fname)
        if fname.endswith(".txt"):
            docs.extend(TextLoader(path, encoding="utf-8").load())
        elif fname.endswith(".pdf"):
            docs.extend(PyPDFLoader(path).load())

    if not docs:
        return None, "no_docs"

    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(os.path.dirname(FAISS_PATH), exist_ok=True)
    vectorstore.save_local(FAISS_PATH)

    return vectorstore, "built"


# ─────────────────────────────────────────────
# 3. LANGGRAPH NODES
# ─────────────────────────────────────────────
def retrieve_node(state, vectorstore):
    docs = vectorstore.as_retriever(search_kwargs={"k": 4}).invoke(state["question"])
    return {**state, "retrieved_docs": docs, "has_relevant_docs": len(docs) > 0}

def grade_node(state):
    stop = {"what","is","are","the","a","an","of","in","how","does","do","tell","me","about"}
    keywords = set(state["question"].lower().split()) - stop
    relevant = any(kw in doc.page_content.lower() for doc in state["retrieved_docs"] for kw in keywords)
    return {**state, "has_relevant_docs": relevant}

def build_context_node(state):
    return {**state, "context": "\n\n---\n\n".join(d.page_content for d in state["retrieved_docs"])}

def generate_node(state, llm):
    prompt = f"""You are a helpful AI assistant. Use the context below to answer the question.
If the context lacks enough info, say so.

Context:
{state['context']}

Question: {state['question']}
Answer:"""
    return {**state, "answer": llm.invoke(prompt).content}

def fallback_node(state, llm):
    prompt = f"""You are a helpful AI assistant. No relevant documents were found.
Answer from general knowledge and mention no documents were retrieved.

Question: {state['question']}
Answer:"""
    return {**state, "answer": llm.invoke(prompt).content}

def route_after_grade(state):
    return "build_context" if state["has_relevant_docs"] else "fallback"


# ─────────────────────────────────────────────
# 4. BUILD LANGGRAPH
# ─────────────────────────────────────────────
def build_graph(vectorstore, llm):
    g = StateGraph(RAGState)
    g.add_node("retrieve",      lambda s: retrieve_node(s, vectorstore))
    g.add_node("grade",         grade_node)
    g.add_node("build_context", build_context_node)
    g.add_node("generate",      lambda s: generate_node(s, llm))
    g.add_node("fallback",      lambda s: fallback_node(s, llm))
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", route_after_grade, {
        "build_context": "build_context",
        "fallback": "fallback"
    })
    g.add_edge("build_context", "generate")
    g.add_edge("generate", END)
    g.add_edge("fallback", END)
    return g.compile()


# ─────────────────────────────────────────────
# 5. STREAMLIT UI
# ─────────────────────────────────────────────
st.set_page_config(page_title="LangGraph RAG", page_icon="🧠", layout="wide")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not found. Add it to your .env file.")
    st.code("OPENAI_API_KEY=sk-your-key-here")
    st.stop()

st.title("🧠 LangGraph RAG Chatbot")
st.caption("Built with LangChain + LangGraph | GPT-4o-mini")

# ── Session state init ──
if "messages"       not in st.session_state: st.session_state.messages = []
if "graph"          not in st.session_state: st.session_state.graph = None
if "uploaded_files" not in st.session_state: st.session_state.uploaded_files = set()

# ── Sidebar ──
with st.sidebar:
    st.subheader("📂 Upload Documents")
    uploaded_file = st.file_uploader("Add a PDF or TXT", type=["pdf", "txt"])

    if uploaded_file and uploaded_file.name not in st.session_state.uploaded_files:
        # Save file to data/
        os.makedirs(DATA_DIR, exist_ok=True)
        save_path = os.path.join(DATA_DIR, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.read())

        # Mark as processed so we don't rerun on next render
        st.session_state.uploaded_files.add(uploaded_file.name)

        # Clear FAISS index so it rebuilds with new file
        if os.path.exists(FAISS_PATH):
            shutil.rmtree(os.path.dirname(FAISS_PATH))
        st.cache_resource.clear()
        st.session_state.graph = None
        st.success(f"✅ {uploaded_file.name} added — rebuilding index...")
        st.rerun()

    st.divider()
    st.subheader("🗺️ LangGraph Flow")
    st.markdown("""
```
[retrieve]
    ↓
[grade]
    ↓ conditional
  ┌──────────────┐
  ↓              ↓
[build_context] [fallback]
  ↓
[generate]
  ↓
 END
```
""")
    st.divider()
    st.subheader("📄 Knowledge Base Files")
    if os.path.exists(DATA_DIR):
        for f in os.listdir(DATA_DIR):
            st.markdown(f"- `{f}`")
    else:
        st.warning("No data/ folder found.")

# ── Build vectorstore & graph ──
vectorstore, vs_status = build_vectorstore()

if vs_status == "no_docs":
    st.error("No documents in data/ folder. Upload a file or add one manually.")
    st.stop()

if st.session_state.graph is None:
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0)
    st.session_state.graph = build_graph(vectorstore, llm)

# ── Display chat history ──
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"📚 Sources ({len(msg['sources'])} chunks)"):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(f"**Chunk {i}:**")
                    st.text(src[:400] + "..." if len(src) > 400 else src)
                    st.divider()

# ── Chat input ──
if question := st.chat_input("Ask anything about the knowledge base..."):
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        with st.spinner("🔄 Running LangGraph pipeline..."):
            result = st.session_state.graph.invoke({
                "question": question,
                "retrieved_docs": [],
                "context": "",
                "answer": "",
                "has_relevant_docs": False
            })

        st.markdown(result["answer"])

        sources = [d.page_content for d in result["retrieved_docs"]]
        if sources:
            with st.expander(f"📚 Sources ({len(sources)} chunks)"):
                for i, src in enumerate(sources, 1):
                    st.markdown(f"**Chunk {i}:**")
                    st.text(src[:400] + "..." if len(src) > 400 else src)
                    st.divider()

        path = "✅ Retrieved from knowledge base" if result["has_relevant_docs"] else "⚠️ Fallback (no relevant docs)"
        st.caption(f"Graph path: {path}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": sources
    })