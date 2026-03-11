import logging
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv, find_dotenv

from ..document_loader import SupportDocumentLoader
from ..vector_store import SupportVectorStore
from ..rag_chain import SupportRAGChain

load_dotenv(find_dotenv())

logger = logging.getLogger(__name__)

class SupportEngine:
    def __init__(
        self,
        data_path: str = "data",
        persist_directory: str = "vector_store",
        openai_api_key: Optional[str] = None
    ):
        self.data_path = data_path
        self.persist_directory = persist_directory
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.vector_store = None
        self.rag_chain = None
        self._initialized = False

    def initialize(self, force_reload: bool = True):
        """Initialize the support engine by loading documents and creating vector store."""
        try:
            logger.info("Initializing Support Engine...")
            
            # Initialize vector store
            self.vector_store = SupportVectorStore(
                persist_directory=self.persist_directory,
                openai_api_key=self.openai_api_key
            )
            
            # Check if we need to load documents
            persist_path = Path(self.persist_directory)
            if True or force_reload or not persist_path.exists() or not any(persist_path.iterdir()):
                logger.info("Loading documents from data directory...")
                loader = SupportDocumentLoader(data_path=self.data_path)
                documents = loader.create_documents()
                
                # Add documents to vector store
                for support_type, docs in documents.items():
                    if docs:
                        logger.info(f"Adding {len(docs)} documents for {support_type} support")
                        self.vector_store.add_documents(docs, support_type)
                
                logger.info("Documents loaded successfully")
            else:
                logger.info("Using existing vector store")
            
            # Initialize RAG chain
            self.rag_chain = SupportRAGChain(vector_store=self.vector_store)
            self._initialized = True
            
            logger.info("Support Engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing Support Engine: {e}")
            raise

    def is_initialized(self) -> bool:
        """Check if the engine is initialized."""
        return self._initialized

    def get_rag_chain(self) -> SupportRAGChain:
        """Get the RAG chain instance."""
        if not self._initialized:
            raise RuntimeError("Engine not initialized. Call initialize() first.")
        return self.rag_chain

    def get_vector_store(self) -> SupportVectorStore:
        """Get the vector store instance."""
        if not self._initialized:
            raise RuntimeError("Engine not initialized. Call initialize() first.")
        return self.vector_store
