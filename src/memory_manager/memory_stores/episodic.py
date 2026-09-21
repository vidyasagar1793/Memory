# episodic_memory.py
import chromadb
import logging
from typing import Optional, List

logger = logging.getLogger("EpisodicMemory")

class EpisodicMemory:
    def __init__(self, collection_name: str = "agent_episodes"):
        """
        Stores full execution trajectories and reflections from past runs.
        """
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def log_episode(
        self, 
        episode_id: str, 
        task: str, 
        success: bool, 
        reflection: str,
        tool_sequence: List[str]
    ):
        """
        Logs a completed task episode into vector memory.
        """
        logger.info(f"Logging episode '{episode_id}' (Success: {success})")
        
        # We store the task + reflection as the embeddable text
        document = f"Task: {task}\nOutcome: {'SUCCESS' if success else 'FAILURE'}\nReflection/Lesson: {reflection}"
        
        metadata = {
            "task": task,
            "success": str(success),
            "tools_used": ",".join(tool_sequence)
        }
        
        self.collection.add(
            documents=[document],
            metadatas=[metadata],
            ids=[episode_id]
        )

    def recall_relevant_lessons(self, current_task: str, n_results: int = 2) -> str:
        """
        Searches for past experiences relevant to the incoming task.
        """
        if self.collection.count() == 0:
            return ""

        logger.info(f"Recalling past episodes for task: '{current_task}'")
        results = self.collection.query(
            query_texts=[current_task],
            n_results=n_results
        )

        if not results['documents'] or not results['documents'][0]:
            return ""

        lessons = []
        for i, doc in enumerate(results['documents'][0]):
            lessons.append(f"--- Past Experience {i+1} ---\n{doc}")

        return "\n\n".join(lessons)