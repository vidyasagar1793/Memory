# main_episodic_demo.py
import logging
import uuid
from memory_manager.agent import Tool, ReActAgent
from memory_manager.semantic_memory import SemanticMemory
from memory_manager.episodic_memory import EpisodicMemory

    
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EpisodicMemoryDemo")

if __name__ == "__main__":
    # Initialize Memory Modules
    semantic_mem = SemanticMemory()
    episodic_mem = EpisodicMemory()

    # 1. Seed Episodic Memory with a past failure/lesson
    # (In production, this is saved dynamically when a task fails)
    episodic_mem.log_episode(
        episode_id=str(uuid.uuid4()),
        task="Query corporate database for user passwords",
        success=False,
        reflection="Attempting to search directly with generic terms like 'passwords' failed. Must search using exact system names (e.g., 'server_passwords').",
        tool_sequence=["SearchMemory"]
    )

    # 2. Incoming Task
    new_task = "Retrieve credentials for the backend server."

    # 3. Recall lessons relevant to new_task
    past_lessons = episodic_mem.recall_relevant_lessons(new_task)

    # 4. Formulate tools
    memory_tool = Tool(
        name="SearchMemory",
        description="Searches long-term knowledge base. Input should be a query string.",
        func=semantic_mem.search
    )

    # 5. Initialize agent and inject lessons into System Memory before execution
    agent = ReActAgent(tools=[memory_tool])
    
    if past_lessons:
        logger.info("Injecting recalled lessons into system prompt...")
        lesson_prompt = f"\n\n[PAST LESSONS FROM SIMILAR TASKS]\n{past_lessons}\nUse these lessons to avoid repeating past mistakes."
        # Append lessons to current system prompt
        agent.memory.history[0].content += lesson_prompt

    # 6. Run the agent
    print("\nStarting execution with Episodic memory active...")
    result = agent.run(new_task)
    print(f"\nFinal Result: {result}")
