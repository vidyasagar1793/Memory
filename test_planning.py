# test_planning.py
from agent import Tool
from semantic_memory import SemanticMemory
from plan_executor2 import PlanExecutor

if __name__ == "__main__":
    # Seed semantic memory with data
    mem = SemanticMemory()
    mem.add_knowledge(
        doc_id="budget_2026",
        text="The infrastructure budget for Coimbatore city expansion in 2026 is 450000000 INR. Allocation for road development is 40% of total budget."
    )

    def math_calc(expr: str) -> str:
        return str(eval(expr))

    search_tool = Tool(
        name="SearchMemory",
        description="Searches long-term knowledge base for facts.",
        func=mem.search
    )
    calc_tool = Tool(
        name="Calculator",
        description="Evaluates math expressions.",
        func=math_calc
    )

    executor = PlanExecutor(tools=[search_tool, calc_tool])
    
    # High-level multi-step request
    goal = "Find the total 2026 Coimbatore infrastructure budget from memory, calculate the exact amount allocated for road development in INR, and summarize the final numbers."
    
    final_output = executor.execute_goal(goal)