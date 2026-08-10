# test_hierarchical.py
import os
from agent import Tool
from semantic_memory import SemanticMemory
from hierarchical_architecture import HierarchicalAgentSystem

if __name__ == "__main__":
    # Ensure GEMINI_API_KEY is configured
    # os.environ["GEMINI_API_KEY"] = "your_key_here"

    # 1. Seed Vector Memory
    mem = SemanticMemory()
    mem.add_knowledge(
        doc_id="coimbatore_infra_2026",
        text="The total infrastructure budget for Coimbatore in 2026 is 4,500,000,000 INR (450 Crore INR). Road development is allocated 40%, public transit receives 35%, and smart water management receives 25%."
    )

    # 2. Toolkits
    search_tool = Tool(
        name="SearchMemory",
        description="Searches long-term vector database for factual infrastructure documents.",
        func=mem.search
    )
    
    # 3. Instantiate Hierarchical System
    # Both workers retrieve the source facts from memory before responding.
    system = HierarchicalAgentSystem(
        research_tools=[search_tool],
        analysis_tools=[search_tool],
        provider="gemini"
    )

    # 4. Execute complex goal
    goal = "Retrieve the 2026 Coimbatore infrastructure budget details from memory, calculate the exact breakdown in INR for roads, transit, and water, and produce a formal financial summary."
    
    report = system.run(goal)
    print("\n================ FINAL SYNTHESIZED REPORT ================")
    print(report)
