# src/search/publication_qa_tool.py
import os
import pickle
import re
from typing import Optional

import streamlit as st
import requests
import pangaeapy.pandataset as pdataset
from pydantic import BaseModel, Field

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# LangChain v1: legacy chains/memory live in langchain-classic
try:
    from langchain.chains import ConversationalRetrievalChain
except Exception:
    from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain

try:
    from langchain.memory import ConversationBufferMemory
except Exception:
    from langchain.memory.buffer import ConversationBufferMemory

# ParentDocumentRetriever still comes from langchain (v1)
try:
    from langchain.retrievers import ParentDocumentRetriever
except Exception:
    from langchain.retrievers.parent_document_retriever import ParentDocumentRetriever



# InMemoryStore moved around between versions; prefer core
try:
    from langchain_core.stores import InMemoryStore
except Exception:
    from langchain.storage import InMemoryStore

from ..config import API_KEY
from ..llm_factory import get_llm


class PublicationQAArgs(BaseModel):
    doi: str = Field(
        description="The DOI of the dataset, e.g., 'https://doi.org/10.1594/PANGAEA.xxxxxx'; make sure to get correct doi, based on the history of messages"
    )
    question: str = Field(
        description="The question to ask about the publication related to the dataset. Please modify the original question of the user! The question should be reworded to specifically send it to RAG. I.e. the original user question 'Are there any related articles to the first dataset? If so what these articles are about?' will be reworded for this tool as 'What is this article about?' Always add at the end to give extended response with great depth and clarity."
    )


def _publication_db_dir() -> str:
    base = os.path.join(os.getcwd(), "data", "publication_database")
    os.makedirs(base, exist_ok=True)
    return base


def get_related_publication_info(doi: str) -> Optional[str]:
    try:
        m = re.search(r"PANGAEA\.(\d+)", doi, flags=re.IGNORECASE)
        if not m:
            return None
        dataset_id = int(m.group(1))

        ds = pdataset.PanDataSet(dataset_id)

        supplement_to = getattr(ds, "supplement_to", None)
        if supplement_to and isinstance(supplement_to, dict) and "uri" in supplement_to:
            related_doi = supplement_to["uri"].split("https://doi.org/")[-1].strip()
            return related_doi or None

        citation = getattr(ds, "citation", "") or ""
        if "In supplement to:" in citation:
            supplement_part = citation.split("In supplement to:", 1)[-1]
            doi_match = re.search(
                r"(?:https?://)?(?:dx\.)?doi\.org/([^\s]+)",
                supplement_part,
                flags=re.IGNORECASE,
            )
            if doi_match:
                return doi_match.group(1).strip()

        return None

    except Exception as e:
        print(f"Error fetching related publication: {str(e)}")
        return None


def create_pdf_filename(doi: Optional[str]) -> Optional[str]:
    if not doi:
        return None
    safe = re.sub(r"[^\w\-.]+", "_", doi.strip())
    return f"{safe}.pdf"


def download_pdf_from_crossref(doi: str) -> Optional[str]:
    crossref_url = f"https://api.crossref.org/works/{doi}"
    headers = {
        "User-Agent": "pangaeaGPT/1.0",
        "Accept": "application/json",
    }

    try:
        print(f"Crossref URL: {crossref_url}")

        response = requests.get(crossref_url, headers=headers, timeout=20)
        response.raise_for_status()
        data = response.json()

        pdf_url = None
        message = (data or {}).get("message") or {}

        links = message.get("link") or []
        if links:
            pdf_url = next((link.get("URL") for link in links if (link.get("URL") or "").lower().endswith(".pdf")), None)

            if not pdf_url:
                pdf_url = next(
                    (
                        link.get("URL")
                        for link in links
                        if link.get("URL")
                        and link.get("content-type") in ("application/pdf", "unspecified")
                        and link.get("intended-application") in ("similarity-checking", "text-mining")
                    ),
                    None,
                )

        if not pdf_url:
            resource = message.get("resource") or {}
            primary = (resource.get("primary") or {}).get("URL")
            if primary and str(primary).lower().endswith(".pdf"):
                pdf_url = primary

        if not pdf_url:
            return None

        print(f"PDF URL: {pdf_url}")

        pdf_response = requests.get(pdf_url, headers={"User-Agent": headers["User-Agent"]}, timeout=40)
        pdf_response.raise_for_status()

        safe_filename = create_pdf_filename(doi)
        if not safe_filename:
            return None

        publication_database = _publication_db_dir()
        pdf_path = os.path.join(publication_database, safe_filename)

        with open(pdf_path, "wb") as f:
            f.write(pdf_response.content)

        print(f"PDF downloaded to: {pdf_path}")
        return pdf_path

    except Exception as e:
        print(f"Error downloading PDF: {str(e)}")
        return None


def save_to_pickle(obj, filename: str) -> None:
    with open(filename, "wb") as file:
        pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)


def load_from_pickle(filename: str):
    with open(filename, "rb") as file:
        return pickle.load(file)


def create_embeddings(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)

    store = InMemoryStore()
    embeddings = OpenAIEmbeddings(api_key=API_KEY)

    chroma_path = pdf_path.replace(".pdf", "_chroma")
    vectorstore = Chroma(
        collection_name="full_documents",
        embedding_function=embeddings,
        persist_directory=chroma_path,
    )

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    retriever.add_documents(documents)

    docstore_path = pdf_path.replace(".pdf", "_docstore.pkl")

    # InMemoryStore in core typically stores data in .store (dict-like).
    store_dict = getattr(retriever.docstore, "store", None)
    if store_dict is None:
        store_dict = {}

    save_to_pickle(store_dict, docstore_path)
    return chroma_path, docstore_path


def load_retriever(docstore_path: str, chroma_path: str):
    embeddings = OpenAIEmbeddings(api_key=API_KEY)
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)

    vectorstore = Chroma(
        collection_name="full_documents",
        embedding_function=embeddings,
        persist_directory=chroma_path,
    )

    store_dict = load_from_pickle(docstore_path)
    store = InMemoryStore()
    if isinstance(store_dict, dict) and store_dict:
        store.mset(list(store_dict.items()))

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )
    return retriever


def answer_publication_questions(doi: str, question: str):
    related_doi = get_related_publication_info(doi)
    if not related_doi:
        return "No publications related to this dataset were found."

    publication_database = _publication_db_dir()

    pdf_filename = create_pdf_filename(related_doi)
    if not pdf_filename:
        return "Unable to build a safe filename for the related publication DOI."

    pdf_path = os.path.join(publication_database, pdf_filename)
    chroma_path = pdf_path.replace(".pdf", "_chroma")
    docstore_path = pdf_path.replace(".pdf", "_docstore.pkl")

    try:
        if not os.path.exists(chroma_path) or not os.path.exists(docstore_path):
            if not os.path.exists(pdf_path):
                downloaded_pdf_path = download_pdf_from_crossref(related_doi)
                if not downloaded_pdf_path:
                    return "Unable to download the related publication PDF."
                pdf_path = downloaded_pdf_path
                chroma_path = pdf_path.replace(".pdf", "_chroma")
                docstore_path = pdf_path.replace(".pdf", "_docstore.pkl")

            chroma_path, docstore_path = create_embeddings(pdf_path)

        retriever = load_retriever(docstore_path, chroma_path)

        llm = get_llm()

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
        )

        response = conversation_chain({"question": question})
        if isinstance(response, dict):
            return response.get("answer") or response.get("result") or str(response)
        return str(response)

    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        return f"An error occurred while processing your request: {str(e)}"
