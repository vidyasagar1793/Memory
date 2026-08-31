# core_models.py
from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class Role(str, Enum):
    """Defines the actors in our agentic system."""
    SYSTEM = "system"       # Instructions and rules for the agent
    USER = "user"           # Human inputs
    ASSISTANT = "assistant" # The LLM's responses (Thoughts/Actions)
    TOOL = "tool"           # Outputs from executed tools/functions

class Message(BaseModel):
    """A standardized unit of communication in our architecture."""
    role: Role
    content: str
    name: Optional[str] = Field(
        default=None, 
        description="Used primarily for Tool roles to identify which tool was called."
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Hidden state data (e.g., token counts, timestamps) not passed to the LLM."
    )

    def to_llm_format(self) -> dict:
        """Converts the Pydantic model into standard OpenAI/Gemini dictionary format."""
        msg = {"role": self.role.value, "content": self.content}
        if self.name:
            msg["name"] = self.name
        return msg