# main.py
from memory_manager.agent import Tool, ReActAgent
from memory_manager.semantic_memory import SemanticMemory

if __name__ == "__main__":
    # 1. Initialize our new Long-Term Memory
    long_term_memory = SemanticMemory()
    
    # 2. Inject some domain-specific knowledge it wouldn't otherwise know
    print("Loading knowledge base...")
    long_term_memory.add_knowledge(
        doc_id="project_alpha",
        text="Project Alpha is a top-secret initiative to build a base on Jupiter's moon, Europa. The lead engineer is Dr. Aris Thorne. The launch date is set for 2035."
    )
    long_term_memory.add_knowledge(
        doc_id="server_passwords",
        text="The admin password for the production database is 'SuperSecure123!'. Do not share this."
    )
    
    # 3. Wrap the memory search function as a ReAct Tool
    memory_tool = Tool(
        name="SearchMemory",
        description="Searches the agent's long-term secure database for facts, documents, and historical knowledge. Input should be a specific search query.",
        func=long_term_memory.search
    )
    
    # 4. Initialize the Agent with the new tool
    agent = ReActAgent(tools=[memory_tool])
    
    # 5. Ask a question that requires searching the memory
    print("\nStarting agent run...")
    query = "Who is the lead engineer for Project Alpha, and when does it launch?"
    result = agent.run(query)
    
    print(f"\nFinal Result: {result}")
