# rag_core.py (LOCAL FILE VERSION — builds dementiaDB if missing)

import os
from dotenv import load_dotenv, find_dotenv
import streamlit as st

from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate

# ------------------------
# LOAD ENV
# ------------------------
load_dotenv(find_dotenv("dotenvKey.env"))
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY missing in dotenvKey.env")


# ------------------------
# LOAD LOCAL DOCS
# ------------------------

def load_local_docs():
    """
    Load all .txt, .pdf and .docx files from ./data into LangChain Documents.
    """
    docs_dir = "data"

    if not os.path.exists(docs_dir):
        raise ValueError(
            "No 'dementiaDB' found and no 'data' folder with source docs.\n"
            "Create a 'data' folder in the project root and add your dementia documents (.pdf, .docx, .txt)."
        )

    docs = []

    for root, _, files in os.walk(docs_dir):
        for fname in files:
            path = os.path.join(root, fname)
            lower = fname.lower()

            try:
                if lower.endswith(".txt"):
                    loader = TextLoader(path, encoding="utf-8")
                elif lower.endswith(".pdf"):
                    loader = PyPDFLoader(path)
                elif lower.endswith(".docx"):
                    loader = Docx2txtLoader(path)
                else:
                    # ignore unsupported file types
                    continue

                docs.extend(loader.load())

            except Exception as e:
                # Skip files that cannot be read, but don't crash
                print(f"⚠️ Skipping {path}: {e}")

    if not docs:
        raise ValueError(
            "Found 'data' folder but no readable .txt/.pdf/.docx documents.\n"
            "Add some dementia-related documents into the 'data' folder."
        )

    return docs


# ------------------------
# VECTORSTORE
# ------------------------

@st.cache_resource
def load_vectorstore():
    """
    Load existing Chroma DB from 'dementiaDB'.
    If it doesn't exist, build it once from documents in ./data.
    """
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    persist_dir = "dementiaDB"

    # 1) Fast path: existing DB
    if os.path.exists(persist_dir):
        return Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
        )

    # 2) Build DB from local docs
    docs = load_local_docs()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )
    chunks = splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=persist_dir,
    )

    # saved automatically in persist_dir
    return vectorstore


vectorstore = load_vectorstore()


# ------------------------
# PROMPT
# ------------------------

template = """
You are a dementia-care assistant.

Use the provided context ONLY for factual information.
If the context does not answer the question, but the user is asking for:
- formatting (table, bullet points, summary, rewrite, simplify)
- restructuring your previous answer
- converting your last answer into another format

THEN you may answer WITHOUT using the context.

Only say "Sorry, I don't know. Please consult a medical professional." when:
- the user asks for factual information
- AND the context does not contain the requested facts.

Context:
{context}

Question:
{question}

Now provide the best possible answer.
"""

qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=template,
)


# ------------------------
# LLM
# ------------------------

@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.3,
        google_api_key=GOOGLE_API_KEY,
    )

llm = load_llm()


# ------------------------
# RAG CHAIN
# ------------------------

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type="stuff",
    chain_type_kwargs={"prompt": qa_prompt},
    return_source_documents=False,
)


# ------------------------
# FUNCTION STREAMLIT CALLS
# ------------------------

def rag_answer(user_message: str, history=None):
    """
    Build a short conversation history string and query the RAG chain.
    """
    try:
        full_question = ""

        # include last few turns
        if history:
            for msg in history[-6:]:
                role = msg["role"].capitalize()
                content = msg["content"]
                full_question += f"{role}: {content}\n"

        full_question += f"User: {user_message}"

        return qa_chain.run(full_question)

    except Exception as e:
        return f"Error: {e}"
