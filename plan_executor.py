# plan_executor.py
import logging
from typing import List
from agent import Tool
from agent import ReActAgent
from planner import Planner
from planner_models import TaskPlan, StepStatus

logger = logging.getLogger("PlanExecutor")

class PlanExecutor:
    def __init__(self, tools: List[Tool]):
        self.tools = tools
        self.planner = Planner(tools=tools)

    def execute_goal(self, goal: str) -> str:
        plan = self.planner.generate_plan(goal)
        print("\n" + plan.get_summary() + "\n")

        accumulated_context = []

        # Phase 2: Execute Plan Step by Step
        for step in plan.steps:
            step.status = StepStatus.IN_PROGRESS
            logger.info(f"Executing Step {step.step_id}: {step.title}")

            # Construct focused prompt with cumulative state from earlier steps
            context_str = "\n".join(accumulated_context)
            step_prompt = (
                f"Subtask Instruction: {step.instruction}\n"
                f"Previous Step Results:\n{context_str if context_str else 'None'}\n"
                f"Complete this subtask and state the final result."
            )

            # Spawn dedicated ReAct sub-agent for this step
            sub_agent = ReActAgent(tools=self.tools, max_iterations=4)
            result = sub_agent.run(step_prompt)

            step.result = result
            step.status = StepStatus.COMPLETED
            accumulated_context.append(f"Step {step.step_id} ({step.title}): {result}")

            print(f"✓ Step {step.step_id} Completed: {result}\n")

        print("=== Execution Complete ===")
        print(plan.get_summary())
        return accumulated_context[-1] if accumulated_context else "No result generated."