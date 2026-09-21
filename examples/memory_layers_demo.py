import logging
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("MemoryManager")

# ==============================================================================
# LAYER 1: SHORT-TERM WORKING MEMORY (Scratchpad per agent)
# ==============================================================================
class ShortTermMemory:
    """Manages the volatile, high-density message thread for a single agent run."""
    def __init__(self):
        self.messages: List[Dict[str, str]] = []

    def add(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})

    def get_context(self) -> str:
        return "\n".join([f"{m['role'].upper()}: {m['content']}" for m in self.messages])

    def clear(self):
        """Wipes internal thoughts/tool traces after task completion to prevent context bloat."""
        self.messages.clear()
        logger.info("Short-Term Memory wiped.")


# ==============================================================================
# LAYER 2: LONG-TERM VECTOR MEMORY (System-wide persistence)
# ==============================================================================
class LongTermMemory:
    """Mock Vector Database representing RAG retrieval."""
    def __init__(self):
        self.store = {
            "coimbatore_infra_2026": "Coimbatore 2026 budget: Total ₹4.5B. Roads=40%, Transit=35%, Water=25%."
        }

    def retrieve(self, query: str) -> str:
        logger.info(f"Retrieving from Long-Term Memory for query: '{query}'")
        for key, value in self.store.items():
            if any(term in query.lower() for term in ["budget", "coimbatore", "infra"]):
                return value
        return "No relevant records found."


# ==============================================================================
# LAYER 3: INTER-AGENT SHARED STATE (Distillation & Handoff)
# ==============================================================================
class SharedWorkflowState:
    """Holds distilled context passed sequentially between specialized workers."""
    def __init__(self):
        self.distilled_outputs: Dict[str, str] = {}

    def save_output(self, worker_name: str, distilled_result: str):
        self.distilled_outputs[worker_name] = distilled_result
        logger.info(f"Saved distilled output from [{worker_name}].")

    def get_accumulated_context(self) -> str:
        if not self.distilled_outputs:
            return "No previous worker context available."
        return "\n".join([f"[{role}]: {res}" for role, res in self.distilled_outputs.items()])


# ==============================================================================
# WORKFLOW ORCHESTRATOR
# ==============================================================================
class BaseWorker:
    def __init__(self, name: str, long_term_mem: LongTermMemory):
        self.name = name
        self.short_term_mem = ShortTermMemory()
        self.long_term_mem = long_term_mem

    def execute(self, task: str, shared_state: SharedWorkflowState) -> str:
        raise NotImplementedError


class ResearchWorker(BaseWorker):
    def execute(self, task: str, shared_state: SharedWorkflowState) -> str:
        # Step 1: Inject Task & Shared Context into Working Memory
        context = shared_state.get_accumulated_context()
        self.short_term_mem.add("system", f"Previous Context:\n{context}")
        self.short_term_mem.add("user", task)

        # Step 2: Fetch facts from Long-Term Memory
        retrieved_data = self.long_term_mem.retrieve(task)
        self.short_term_mem.add("observation", f"Fetched Data: {retrieved_data}")

        # Step 3: Extract ground truth (Simulated LLM synthesis)
        final_distilled_answer = f"Retrieved Data Point: {retrieved_data}"
        
        # Step 4: Save distilled result to Shared State & Wipe Short-Term Memory
        shared_state.save_output(self.name, final_distilled_answer)
        self.short_term_mem.clear()  # Wipes raw scratchpad, keeps shared state clean
        
        return final_distilled_answer


class AnalysisWorker(BaseWorker):
    def execute(self, task: str, shared_state: SharedWorkflowState) -> str:
        # Step 1: Receives ONLY the clean distilled state from ResearchWorker
        prior_context = shared_state.get_accumulated_context()
        self.short_term_mem.add("system", f"Context from prior workers:\n{prior_context}")
        self.short_term_mem.add("user", task)

        # Step 2: Perform math logic without tool clutter
        calculated_answer = "Public Transit Allocation (35% of 4.5B) = ₹1.575 Billion."
        
        shared_state.save_output(self.name, calculated_answer)
        self.short_term_mem.clear()
        
        return calculated_answer


# ==============================================================================
# EXECUTION TEST
# ==============================================================================
if __name__ == "__main__":
    lt_memory = LongTermMemory()
    workflow_state = SharedWorkflowState()

    researcher = ResearchWorker("ResearchWorker", lt_memory)
    analyst = AnalysisWorker("AnalysisWorker", lt_memory)

    print("\n--- PHASE 1: RESEARCH ---")
    researcher.execute("Retrieve Coimbatore budget 2026", workflow_state)

    print("\n--- PHASE 2: ANALYSIS ---")
    analyst.execute("Calculate 45% public transit allocation", workflow_state)

    print("\n--- FINAL SHARED WORKFLOW STATE ---")
    print(workflow_state.get_accumulated_context())