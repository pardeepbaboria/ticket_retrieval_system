import logging
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any
from langchain_core.documents import Document


logger = logging.getLogger(__name__)

class SupportVectorStore:
    def __init__(self, persist_directory: str, openai_api_key: str):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=openai_api_key,
            model_name="text-embedding-ada-002"
        )
        self.collections = {}

    def _get_or_create_collection(self, support_type: str):
        if support_type not in self.collections:
            self.collections[support_type] = self.client.get_or_create_collection(
                name=f"support_{support_type}",
                embedding_function=self.ef
            )
        return self.collections[support_type]

    def _prepare_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        clean_metadata = {}
        for k, v in metadata.items():
            if isinstance(v, list):
                clean_metadata[k] = ", ".join(v) if v else ""
            elif v is None:
                clean_metadata[k] = ""
            else:
                clean_metadata[k] = v
        return clean_metadata

    def _process_metadata_for_return(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        if 'tags' in metadata and isinstance(metadata['tags'], str):
            metadata['tags'] = [t.strip() for t in metadata['tags'].split(",") if t.strip()]
        return metadata

    def add_documents(self, documents: List[Document], support_type: str):
        if not documents:
            return
        
        collection = self._get_or_create_collection(support_type)
        
        ids = [doc.metadata.get('ticket_id') for doc in documents]
        texts = [doc.page_content for doc in documents]
        metadatas = [self._prepare_metadata(doc.metadata) for doc in documents]
        
        collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas
        )

    def query_similar(self, query: str, support_type: str = None, k: int = 3) -> List[Dict[str, Any]]:
        if not query or not query.strip():
            logger.warning("Empty query received in query_similar")
            return []

        if not support_type or support_type not in ["technical", "product", "customer"]:
            logger.warning(f"Support type '{support_type}' not found")
            return []

        collection = self._get_or_create_collection(support_type)
        
        results = collection.query(
            query_texts=[query],
            n_results=k
        )

        formatted_results = []
        if results['documents']:
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'content': results['documents'][0][i],
                    'metadata': self._process_metadata_for_return(results['metadatas'][0][i]),
                    'id': results['ids'][0][i]
                })
        
        return formatted_results