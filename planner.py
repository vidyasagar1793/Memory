# planner.py
import json
import logging
import os

from typing import List

from huggingface_hub import InferenceClient
from agent import Tool
from planner_models import TaskPlan, SubTask, StepStatus

logger = logging.getLogger("Planner")

PLANNER_PROMPT = """You are an expert Strategic Task Planner.
Decompose the overarching goal into a sequence of concrete, logical subtasks.

Available Tools:
{tool_descriptions}

User Goal: {goal}

Respond STRICTLY with a valid JSON object matching this schema:
{{
  "goal": "{goal}",
  "steps": [
    {{
      "step_id": 1,
      "title": "Short title",
      "instruction": "Detailed instruction for step 1",
      "assigned_tool": "ToolName or null"
    }}
  ]
}}
Do NOT wrap the output in markdown codeblocks (no ```json). Output raw JSON only.
"""

class Planner:
    def __init__(self, tools: List[Tool], max_iterations: int = 5, model: str = "meta-llama/Llama-3.3-70B-Instruct"):
        self.tools = {tool.name: tool for tool in tools}
        self.max_iterations = max_iterations
        self.model = model
        hf_token = os.getenv("HF_TOKEN")
        self.client = InferenceClient(api_key=hf_token)

    def generate_plan(self, goal: str) -> TaskPlan:
        logger.info(f"Generating step-by-step plan for goal: '{goal}'")
        tool_desc = "\n".join([f"- {t.name}: {t.description}" for t in self.tools.values()])
        
        prompt = PLANNER_PROMPT.format(tool_descriptions=tool_desc, goal=goal)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        raw_text = (response.choices[0].message.content or "").strip()
        # Clean potential markdown fences if present
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
            raw_text = raw_text.strip()

        try:
            data = json.loads(raw_text)
            plan = TaskPlan(**data)
            logger.info(f"Plan generated successfully with {len(plan.steps)} steps.")
            return plan
        except Exception as e:
            logger.error(f"Failed to parse plan JSON: {e}\nRaw Output: {raw_text}")
            # Fallback to a single-step default plan
            return TaskPlan(
                goal=goal,
                steps=[SubTask(step_id=1, title="Execute Goal", instruction=goal)]
            )
