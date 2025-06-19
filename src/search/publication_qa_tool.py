#publication_qa_tool.py
import os
import pandas as pd
import pickle
import streamlit as st
import requests
import pangaeapy.pandataset as pdataset
import re
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from pydantic import BaseModel, Field
from ..config import API_KEY
from ..llm_factory import get_llm # Use the new factory


class PublicationQAArgs(BaseModel):
    doi: str = Field(
        description="The DOI of the dataset, e.g., 'https://doi.org/10.1594/PANGAEA.xxxxxx'; make sure to get correct doi, based on the history of messages")
    question: str = Field(
        description="The question to ask about the publication related to the dataset. Please modify the original question of the user! The question should be reworded to specifically send it to RAG. I.e. the original user question 'Are there any related articles to the first dataset? If so what these articles are about?' will be reworded for this tool as 'What is this article about?' Always add at the end to give extended response with great depth and clarity.")


def get_related_publication_info(doi):
    try:
        dataset_id = doi.split('.')[-1]
        ds = pdataset.PanDataSet(int(dataset_id))

        # Check supplement_to first
        supplement_to = ds.supplement_to
        if supplement_to and 'uri' in supplement_to:
            related_doi = supplement_to['uri'].split('https://doi.org/')[-1]
            return related_doi

        # If no supplement_to, check citation
        citation = ds.citation
        if 'In supplement to:' in citation:
            # Extract the part after 'In supplement to:'
            supplement_part = citation.split('In supplement to:')[-1]

            # Look for a DOI pattern
            doi_match = re.search(r'(?:https?://)?(?:dx\.)?doi\.org/(.+?)(?:\s|$)', supplement_part)
            if doi_match:
                return doi_match.group(1)  # Return the DOI without 'https://doi.org/'

        print("No related publication found in supplement_to or citation.")
        return None

    except Exception as e:
        print(f"Error fetching related publication: {str(e)}")
        return None


def create_pdf_filename(doi):
    if doi:
        return re.sub(r"[\/]", "_", doi) + ".pdf"
    return None


def download_pdf_from_crossref(doi):
    crossref_url = f'https://api.crossref.org/works/{doi}'
    try:
        print(f"Crossref URL: {crossref_url}")

        response = requests.get(crossref_url)
        response.raise_for_status()
        data = response.json()

        pdf_url = None
        if 'message' in data and 'link' in data['message']:
            pdf_url = next((link['URL'] for link in data['message']['link']
                            if link.get('content-type') == 'unspecified'
                            and 'intended-application' in link
                            and link['intended-application'] == 'similarity-checking'), None)

            if not pdf_url:
                pdf_url = next((link['URL'] for link in data['message']['link']
                                if link['URL'].endswith('.pdf')), None)

            if not pdf_url and 'resource' in data['message']:
                pdf_url = data['message']['resource'].get('primary', {}).get('URL')

        if pdf_url:
            print(f"PDF URL: {pdf_url}")

            pdf_response = requests.get(pdf_url)
            pdf_response.raise_for_status()

            safe_filename = create_pdf_filename(doi)
            publication_database = os.path.join(os.getcwd(), 'data', 'publication_database')
            os.makedirs(publication_database, exist_ok=True)
            pdf_path = os.path.join(publication_database, safe_filename)

            with open(pdf_path, 'wb') as f:
                f.write(pdf_response.content)

            print(f"PDF downloaded to: {pdf_path}")
            return pdf_path
    except Exception as e:
        print(f"Error downloading PDF: {str(e)}")
    return None


def save_to_pickle(obj, filename):
    with open(filename, "wb") as file:
        pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)


def load_from_pickle(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)


def create_embeddings(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)

    store = InMemoryStore()
    embeddings = OpenAIEmbeddings(api_key=API_KEY)

    chroma_path = pdf_path.replace('.pdf', '_chroma')
    vectorstore = Chroma(collection_name="full_documents",
                         embedding_function=embeddings,
                         persist_directory=chroma_path)

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter
    )

    retriever.add_documents(documents)

    docstore_path = pdf_path.replace('.pdf', '_docstore.pkl')
    save_to_pickle(retriever.docstore.store, docstore_path)

    return chroma_path, docstore_path


def load_retriever(docstore_path, chroma_path):
    embeddings = OpenAIEmbeddings(api_key=API_KEY)
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)

    vectorstore = Chroma(collection_name="full_documents",
                         embedding_function=embeddings,
                         persist_directory=chroma_path)

    store_dict = load_from_pickle(docstore_path)
    store = InMemoryStore()
    store.mset(list(store_dict.items()))

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter
    )

    return retriever


def answer_publication_questions(doi: str, question: str):
    related_doi = get_related_publication_info(doi)

    if not related_doi:
        return "No publications related to this dataset were found."

    pdf_filename = create_pdf_filename(related_doi)
    publication_database = os.path.join(os.getcwd(), 'publication_database')
    chroma_path = os.path.join(publication_database, pdf_filename.replace(".pdf", "_chroma"))
    docstore_path = os.path.join(publication_database, pdf_filename.replace(".pdf", "_docstore.pkl"))

    try:
        if not os.path.exists(chroma_path) or not os.path.exists(docstore_path):
            pdf_path = download_pdf_from_crossref(related_doi)

            if not pdf_path:
                return "Unable to download the related publication PDF."

            chroma_path, docstore_path = create_embeddings(pdf_path)

        retriever = load_retriever(docstore_path, chroma_path)
        
        # Use the factory to get the LLM instance
        llm = get_llm()

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory
        )

        response = conversation_chain({"question": question})
        return response['answer']

    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        return f"An error occurred while processing your request: {str(e)}"