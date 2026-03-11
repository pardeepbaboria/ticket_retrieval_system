from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import logging
from .vector_store import SupportVectorStore
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

logger = logging.getLogger(__name__)

class SupportRAGChain:
    def __init__(self, vector_store: SupportVectorStore):
        self.vector_store = vector_store
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.prompt = ChatPromptTemplate.from_template("""
        You are a support assistant. Use the following retrieved tickets to answer the user query.
        If no relevant tickets are found, say "No relevant support tickets found."
        
        Context:
        {context}
        
        Query: {query}
        """)

    def _validate_query(self, query: str):
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        if len(query.strip()) < 10:
            raise ValueError("Query too short. Please provide more details.")

    def get_relevant_documents(self, query: str, support_type: str = None, k: int = 3) -> List[Dict[str, Any]]:
        self._validate_query(query)
        return self.vector_store.query_similar(query, support_type, k)

    def _prepare_context(self, documents: List[Dict[str, Any]]) -> str:
        if not documents:
            return "No relevant support tickets found."
        
        context_blocks = []
        for i, doc in enumerate(documents, 1):
            metadata = doc.get('metadata', {})
            tags = ", ".join(metadata.get('tags', []))
            block = (
                f"Ticket {i}:\n"
                f"Support Type: {metadata.get('support_type', 'Unknown')}\n"
                f"Tags: {tags}\n"
                f"Content: {doc.get('content')}"
            )
            context_blocks.append(block)
        return "\n\n".join(context_blocks)

    async def query(self, query: str, support_type: str = None) -> str:
        self._validate_query(query)
        
        try:
            docs = self.get_relevant_documents(query, support_type)
            context = self._prepare_context(docs)
            
            if context == "No relevant support tickets found.":
                return context

            chain = self.prompt | self.llm
            response = await chain.ainvoke({"context": context, "query": query})
            return response.content
        except Exception as e:
            logger.error(f"LLM Generation Error: {e}")
            raise