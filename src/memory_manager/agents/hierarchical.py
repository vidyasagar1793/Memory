# hierarchical_architecture.py
import json
import logging
from typing import List, Dict
from pydantic import BaseModel, Field
from .agent import Tool
from .agent import ReActAgent
from .llm_client import call_llm

logger = logging.getLogger("HierarchicalPlanner")

class WorkerAssignment(BaseModel):
    worker_role: str = Field(description="Role required: 'ResearchWorker' or 'AnalysisWorker'")
    task_description: str = Field(description="Specific instructions for this worker.")

class MetaPlan(BaseModel):
    high_level_goal: str
    assignments: List[WorkerAssignment]

META_PLANNER_PROMPT = """You are a Meta-Planner Architect. 
Decompose the user's overarching goal into specialized worker assignments.

Available Worker Roles:
1. ResearchWorker: Best for searching vector memory, retrieving facts, and pulling document specs.
2. AnalysisWorker: Best for performing calculations, math evaluations, and data synthesis.

User Goal: {goal}

Respond strictly in JSON matching this schema:
{{
  "high_level_goal": "{goal}",
  "assignments": [
    {{
      "worker_role": "ResearchWorker",
      "task_description": "Detailed task instructions..."
    }}
  ]
}}
"""

class HierarchicalAgentSystem:
    def __init__(self, research_tools: List[Tool], analysis_tools: List[Tool], provider: str | None = None):
        self.provider = provider
        print(f"HierarchicalAgentSystem initialized with provider: {self.provider}")
        # Specialized Worker Agents with restricted toolsets
        self.workers: Dict[str, ReActAgent] = {
            "ResearchWorker": ReActAgent(tools=research_tools),
            "AnalysisWorker": ReActAgent(tools=analysis_tools)
        }

    def _generate_meta_plan(self, goal: str) -> MetaPlan:
        prompt = META_PLANNER_PROMPT.format(goal=goal)
        raw_response = call_llm(prompt, provider=self.provider)

        raw_response = raw_response.strip()
        if raw_response.startswith("```"):
            raw_response = raw_response.split("```", 1)[1]
            if raw_response.lstrip().startswith("json"):
                raw_response = raw_response.lstrip()[4:]
            raw_response = raw_response.split("```", 1)[0].strip()

        try:
            data = json.loads(raw_response)
            return MetaPlan(**data)
        except (json.JSONDecodeError, ValueError) as error:
            logger.error("Meta-planner returned invalid JSON: %r", raw_response)
            raise RuntimeError(
                "The meta-planner did not return a valid JSON plan. "
                "Check the model configuration and its response format."
            ) from error

    def run(self, goal: str) -> str:
        logger.info(f"Hierarchical System starting for goal: '{goal}'")
        
        # 1. Meta-Planning
        meta_plan = self._generate_meta_plan(goal)
        logger.info(f"Meta-Plan generated with {len(meta_plan.assignments)} assignments.")

        worker_results = []

        # 2. Delegate to Specialized Workers
        for idx, assignment in enumerate(meta_plan.assignments, 1):
            role = assignment.worker_role
            task = assignment.task_description
            
            if role not in self.workers:
                logger.warning(f"Unknown role '{role}', defaulting to ResearchWorker.")
                role = "ResearchWorker"

            logger.info(f"--- Assigning Subtask {idx} to [{role}] ---")
            worker = self.workers[role]
            
            # Pass results of previous worker outputs as context
            context_payload = "\n".join(worker_results)
            full_task_prompt = (
               f"Task Instruction: {task}\n"
                f"CRITICAL: You MUST use the SearchMemory tool first to fetch exact values "
               f"before generating any output. Do not guess numbers."
                f"Context from previous workers:\n{context_payload if context_payload else 'None'}"
            )
            
            result = worker.run(full_task_prompt)
            worker_results.append(f"[{role} Output]: {result}")

        # 3. Final Synthesis by Meta-Planner
        synthesis_prompt = (
            f"Original Goal: {goal}\n\n"
            f"Worker Output Traces:\n" + "\n".join(worker_results) + "\n\n"
            f"Synthesize the findings into a clear, comprehensive final report."
        )
        
        logger.info("Synthesizing final worker outputs...")
        final_synthesis = call_llm(synthesis_prompt, provider=self.provider)
        return final_synthesis
