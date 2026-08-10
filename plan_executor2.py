# Update inside plan_executor.py
from asyncio.log import logger

from rpds import List as RPDSList
from typing import List as TypingList

from agent import ReActAgent, Tool
from evaluator import Evaluator
from planner import Planner
from planner_models import StepStatus

class PlanExecutor:
    def __init__(self, tools: TypingList[Tool]):
        self.tools = tools
        self.planner = Planner(tools=tools)
        self.evaluator = Evaluator()

    def execute_goal(self, goal: str) -> str:
        plan = self.planner.generate_plan(goal)
        print("\n" + plan.get_summary() + "\n")

        accumulated_context = []

        for step in plan.steps:
            step.status = StepStatus.IN_PROGRESS
            logger.info(f"Executing Step {step.step_id}: {step.title}")
            
            max_retries = 2
            attempts = 0
            step_success = False
            feedback_context = ""

            while attempts < max_retries and not step_success:
                attempts += 1
                
                context_str = "\n".join(accumulated_context)
                step_prompt = (
                    f"Subtask Instruction: {step.instruction}\n"
                    f"Previous Step Results:\n{context_str if context_str else 'None'}\n"
                    f"{feedback_context}"
                    f"Complete this subtask and state the final result."
                )

                sub_agent = ReActAgent(tools=self.tools, max_iterations=4)
                result = sub_agent.run(step_prompt)

                # EVALUATION PHASE
                eval_result = self.evaluator.evaluate_step(step.instruction, result)
                
                if eval_result.success:
                    step.result = result
                    step.status = StepStatus.COMPLETED
                    step_success = True
                    accumulated_context.append(f"Step {step.step_id}: {result}")
                    print(f"✓ Step {step.step_id} Completed: {result}\n")
                else:
                    logger.warning(f"Step failed evaluation. Feedback: {eval_result.feedback}")
                    feedback_context = f"\nCRITICAL FEEDBACK FROM PREVIOUS ATTEMPT: {eval_result.feedback}. You must correct this.\n"
                    
            if not step_success:
                print(f"✗ Step {step.step_id} Failed after {max_retries} attempts. Aborting plan.")
                step.status = StepStatus.FAILED
                break # Stop executing the rest of the plan!

        print("\n=== Execution Complete ===")
        print(plan.get_summary())
        return accumulated_context[-1] if accumulated_context else "Execution failed."