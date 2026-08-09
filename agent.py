"""A small ReAct agent with a safe calculator tool."""

import ast
import logging
import os
import re
from typing import Callable, List

from huggingface_hub import InferenceClient

from core_models import Message, Role
from memory import ShortTermMemory

logger = logging.getLogger("ReActAgent")

REACT_SYSTEM_PROMPT = """You are a logical reasoning agent. You solve problems by interleaving Thoughts and Actions.
You have access to the following tools:
{tool_descriptions}

Use the following format strictly:
Question: the input question you must answer
Thought: you should always think about what to do next
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!
"""


class Tool:
    """A standard wrapper for a function the agent can use."""

    def __init__(self, name: str, description: str, func: Callable[[str], str]):
        self.name = name
        self.description = description
        self.func = func

    def execute(self, action_input: str) -> str:
        logger.info("Executing tool %s with input %r", self.name, action_input)
        try:
            return str(self.func(action_input))
        except Exception as error:
            return f"Error executing {self.name}: {error}"


_BINARY_OPERATORS = {
    ast.Add: lambda left, right: left + right,
    ast.Sub: lambda left, right: left - right,
    ast.Mult: lambda left, right: left * right,
    ast.Div: lambda left, right: left / right,
    ast.FloorDiv: lambda left, right: left // right,
    ast.Mod: lambda left, right: left % right,
    ast.Pow: lambda left, right: left ** right,
}
_UNARY_OPERATORS = {ast.UAdd: lambda value: value, ast.USub: lambda value: -value}


def calculator(expression: str) -> int | float:
    """Evaluate a basic arithmetic expression without exposing Python eval."""

    def evaluate(node: ast.AST) -> int | float:
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
            return _BINARY_OPERATORS[type(node.op)](evaluate(node.left), evaluate(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
            return _UNARY_OPERATORS[type(node.op)](evaluate(node.operand))
        raise ValueError("Only numeric arithmetic expressions are supported.")

    return evaluate(ast.parse(expression, mode="eval").body)


class ReActAgent:
    def __init__(self, tools: List[Tool], max_iterations: int = 5, model: str = "Qwen/Qwen2.5-7B-Instruct"):
        self.tools = {tool.name: tool for tool in tools}
        self.max_iterations = max_iterations
        self.model = model
        hf_token = os.getenv("HF_TOKEN")
        self.memory = ShortTermMemory(max_tokens=3000)
        self.client = InferenceClient(api_key=hf_token)

        tool_desc = "\n".join(f"- {tool.name}: {tool.description}" for tool in tools)
        self.system_content = REACT_SYSTEM_PROMPT.format(
            tool_descriptions=tool_desc,
            tool_names=", ".join(self.tools),
        )
        self.memory.add(Message(role=Role.SYSTEM, content=self.system_content))

    def _live_llm_call(self, context: List[dict]) -> str:
        try:
            response = self.client.chat.completions.create(model=self.model, messages=context)
            return response.choices[0].message.content or ""
        except Exception as error:
            logger.exception("LLM call failed")
            return f"Thought: The language-model call failed.\nFinal Answer: {error}"

    def run(self, user_query: str) -> str:
        self.memory.clear()
        self.memory.add(Message(role=Role.USER, content=f"Question: {user_query}"))

        for iteration in range(self.max_iterations):
            logger.info("--- Iteration %s ---", iteration + 1)
            llm_response = self._live_llm_call(self.memory.get_context())
            self.memory.add(Message(role=Role.ASSISTANT, content=llm_response))

            if "Final Answer:" in llm_response:
                return llm_response.rsplit("Final Answer:", 1)[1].strip()

            action_match = re.search(r"^Action:\s*(.+?)\s*$", llm_response, re.MULTILINE)
            input_match = re.search(r"^Action Input:\s*(.+?)\s*$", llm_response, re.MULTILINE)
            if not (action_match and input_match):
                self.memory.add(Message(
                    role=Role.USER,
                    content="Observation: Your output must include Thought, Action, and Action Input.",
                ))
                continue

            action_name, action_input = action_match.group(1), input_match.group(1)
            if action_name in self.tools:
                observation = self.tools[action_name].execute(action_input)
            else:
                observation = f"Tool '{action_name}' not found. Choose from: {', '.join(self.tools)}"
            self.memory.add(Message(role=Role.USER, content=f"Observation: {observation}"))

        return "Max iterations reached without finding a final answer."


if __name__ == "__main__":
    agent = ReActAgent([
        Tool("Calculator", "Evaluates a basic arithmetic expression, e.g. 12 * (3 + 4).", calculator)
    ])
    print(agent.run("What is (42 * 17) + 8? Use the Calculator tool."))
