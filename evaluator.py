# evaluator.py
import json
import logging
from pydantic import BaseModel, Field
from llm_client import call_llm, LLMProviderException

logger = logging.getLogger("Evaluator")

class EvaluationResult(BaseModel):
    success: bool = Field(description="True if step instruction was fulfilled successfully.")
    feedback: str = Field(description="Actionable critique or validation.")

EVALUATOR_PROMPT = """You are a QA Evaluator. Inspect the subtask instruction and output.

Instruction: {instruction}
Agent Output: {result}

Determine if the subtask was successfully fulfilled.
Respond strictly in JSON matching this schema:
{{
  "success": true/false,
  "feedback": "string critique"
}}
"""

class Evaluator:
    def __init__(self, provider: str = "gemini"):
        self.provider = provider

    def evaluate_step(self, instruction: str, result: str) -> EvaluationResult:
        logger.info("Evaluating step output...")
        
        # Guardrail: Check for obvious error strings in output
        if "Payment Required" in result or "API Error" in result or "402" in result:
            return EvaluationResult(
                success=False, 
                feedback="Step failed because the underlying LLM provider returned an API error."
            )

        prompt = EVALUATOR_PROMPT.format(instruction=instruction, result=result)
        
        try:
            response_text = call_llm(prompt, provider=self.provider)
            
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            data = json.loads(response_text)
            return EvaluationResult(**data)
        except (LLMProviderException, json.JSONDecodeError) as e:
            logger.error(f"Evaluator check failed: {e}")
            # STRICT SAFETY FIX: Fail the step if evaluation cannot be performed
            return EvaluationResult(
                success=False, 
                feedback=f"Evaluation failed due to parsing or API error: {e}"
            )