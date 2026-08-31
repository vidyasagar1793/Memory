# planner_models.py
from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Optional

class StepStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class SubTask(BaseModel):
    step_id: int = Field(description="The numeric sequence order of the step (1, 2, 3...).")
    title: str = Field(description="Short descriptive name of the subtask.")
    instruction: str = Field(description="Detailed instructions on what needs to be achieved in this step.")
    assigned_tool: Optional[str] = Field(default=None, description="Recommended tool to use for this step, if applicable.")
    status: StepStatus = Field(default=StepStatus.PENDING)
    result: Optional[str] = Field(default=None, description="Observation or answer produced after step execution.")

class TaskPlan(BaseModel):
    goal: str = Field(description="The overarching user goal.")
    steps: List[SubTask] = Field(description="The sequence of subtasks needed to satisfy the goal.")

    def get_summary(self) -> str:
        """Returns a scannable status report of the plan."""
        out = [f"=== Plan for Goal: {self.goal} ==="]
        for s in self.steps:
            icon = "✓" if s.status == StepStatus.COMPLETED else ("✗" if s.status == StepStatus.FAILED else "•")
            out.append(f"[{icon}] Step {s.step_id}: {s.title} ({s.status.value})")
            if s.result:
                out.append(f"    Result: {s.result[:100]}...")
        return "\n".join(out)