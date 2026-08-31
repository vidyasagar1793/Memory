# semantic_memory.py
import chromadb
import logging

logger = logging.getLogger("SemanticMemory")

class SemanticMemory:
    def __init__(self, collection_name: str = "agent_knowledge"):
        """
        Initializes an in-memory vector database. 
        Chroma will automatically download a lightweight embedding model 
        (all-MiniLM-L6-v2) the first time you run this.
        """
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(name=collection_name)
        
    def add_knowledge(self, doc_id: str, text: str, metadata: dict = None):
        """Embeds and stores a piece of knowledge."""
        logger.info(f"Memorizing document: {doc_id}")
        self.collection.add(
            documents=[text],
            metadatas=[metadata] if metadata else [{"source": "user_input"}],
            ids=[doc_id]
        )

    def search(self, query: str, n_results: int = 2) -> str:
        """
        Searches the database for concepts mathematically similar to the query.
        Returns a formatted string for the ReAct loop to read.
        """
        logger.info(f"Querying semantic memory for: '{query}'")
        
        # If the database is empty, return early
        if self.collection.count() == 0:
            return "Observation: Memory is currently empty."
            
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        if not results['documents'] or not results['documents'][0]:
            return "Observation: No relevant information found in memory."
            
        # Format the results so the LLM can easily read them
        retrieved_texts = []
        for i, doc in enumerate(results['documents'][0]):
            retrieved_texts.append(f"[Memory Fragment {i+1}]: {doc}")
            
        return "\n".join(retrieved_texts)