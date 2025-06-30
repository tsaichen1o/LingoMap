# rag_search.py
import chromadb
import logging
from typing import List, Dict
import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration (Based on your vocabulary_processor.py) ---
# These settings must match your online ChromaDB instance.
COLLECTION_NAME = "lingomap_vocab"

class RAGSearcher:
    """
    A class to handle Retrieval-Augmented Generation (RAG) searches
    against an ONLINE ChromaDB vector database using HttpClient.
    """
    def __init__(self):
        """Initializes the ChromaDB HttpClient and gets the collection."""
        chroma_api_key = os.getenv("CHROMADB_API_KEY")
        chroma_tenant = os.getenv("CHROMADB_TENANT_ID")
        chroma_database = os.getenv("CHROMADB_NAME")
        if chroma_api_key and chroma_tenant and chroma_database:
            try:
                # Use HttpClient for connecting to an online/remote ChromaDB instance
                self.client = chromadb.CloudClient(
                    tenant=chroma_tenant,
                    database=chroma_database,
                    api_key=chroma_api_key
                )
                # Verify that the server is alive
                self.client.heartbeat()
                self.collection = self.client.get_collection(name=COLLECTION_NAME)
                logging.info(f"✅ RAG Searcher connected to ONLINE ChromaDB at, collection '{COLLECTION_NAME}'.")
            except Exception as e:
                logging.error(f"❌ Failed to connect to online ChromaDB: {e}")
                st.error(f"Failed to connect to RAG database. Please ensure the ChromaDB server is running.")
                self.collection = None

    def search(self, query_text: str, n_results: int = 5) -> List[str]:
        """
        Performs a similarity search in the vector database.

        Args:
            query_text: The text to search for (e.g., cluster name and column names).
            n_results: The number of results to return.

        Returns:
            A list of the most relevant vocabulary terms (URIs or prefixed names).
        """
        if not self.collection:
            logging.warning("Cannot perform search because ChromaDB collection is not available.")
            return []

        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results
            )
            
            # Ensure metadatas and documents are not None
            metadatas = results.get('metadatas') or [[]]
            documents = results.get('documents') or [[]]

            # Extract the 'uri' from the metadata of the retrieved documents and ensure all are strings
            retrieved_uris = [
                str(meta.get('uri', doc))
                for metadatas, docs in zip(metadatas, documents)
                for meta, doc in zip(metadatas, docs)
            ]

            logging.info(f"RAG search for '{query_text}' returned {len(retrieved_uris)} results.")
            return retrieved_uris
        except Exception as e:
            logging.error(f"An error occurred during RAG search: {e}")
            return []

# Use Streamlit's caching for a singleton instance
@st.cache_resource
def get_rag_searcher():
    """Initializes and returns a single instance of the RAGSearcher."""
    return RAGSearcher()