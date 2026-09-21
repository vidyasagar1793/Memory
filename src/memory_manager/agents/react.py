"""A small ReAct agent with a safe calculator tool."""

import ast
import logging
import re
from typing import Callable, List

from .core_models import Message, Role
from .llm_client import LLMClientError, call_llm
from .memory import ShortTermMemory

logger = logging.getLogger("ReActAgent")

REACT_SYSTEM_PROMPT = """You are a logical reasoning agent. You solve problems by interleaving Thoughts and Actions.
You have access to the following tools:
{tool_descriptions}

CRITICAL RULES:
1. If you do not have the information in your context, you MUST use a tool. Do NOT answer from imagination or say data is missing without searching first.
2. Follow this EXACT format:

Question: the input question
Thought: analyze what information is missing
Action: the action to take, must be one of [{tool_names}]
Action Input: the tool argument
Observation: result of the action
... (repeat Thought/Action/Action Input/Observation if needed)
Thought: I now have the answer
Final Answer: the final answer

--- EXAMPLE OF CORRECT TOOL USE ---
Question: What is the server port for DB2?
Thought: I need to find the server port for DB2 in long-term memory.
Action: SearchMemory
Action Input: DB2 server port
Observation: [Memory Fragment 1]: DB2 server runs on port 50000.
Thought: I now know the port number.
Final Answer: The server port for DB2 is 50000.
--- END EXAMPLE ---

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
    def __init__(
        self,
        tools: List[Tool],
        max_iterations: int = 5,
        provider: str | None = None,
        model: str = "llama-3.3-70b-versatile",
    ):
        self.tools = {tool.name: tool for tool in tools}
        self.max_iterations = max_iterations
        self.provider = provider
        self.model = model
        self.memory = ShortTermMemory(max_tokens=3000)

        tool_desc = "\n".join(f"- {tool.name}: {tool.description}" for tool in tools)
        self.system_content = REACT_SYSTEM_PROMPT.format(
            tool_descriptions=tool_desc,
            tool_names=", ".join(self.tools),
        )
        self.memory.add(Message(role=Role.SYSTEM, content=self.system_content))

    def _live_llm_call(self, context: List[dict]) -> str:
        try:
            return call_llm(messages=context, provider=self.provider, model=self.model)
        except LLMClientError as error:
            logger.exception("LLM call failed")
            return f"Thought: The language-model call failed.\nFinal Answer: {error}"

    def run(self, user_query: str) -> str:
        self.memory.clear()
        self.memory.add(Message(role=Role.USER, content=f"Question: {user_query}"))
        tool_executed = False

        for iteration in range(self.max_iterations):
            logger.info("--- Iteration %s ---", iteration + 1)
            llm_response = self._live_llm_call(self.memory.get_context())
            self.memory.add(Message(role=Role.ASSISTANT, content=llm_response))

            action_match = re.search(r"^Action:\s*(.+?)\s*$", llm_response, re.MULTILINE)
            input_match = re.search(r"^Action Input:\s*(.+?)\s*$", llm_response, re.MULTILINE)
            if action_match and input_match:
                action_name, action_input = action_match.group(1), input_match.group(1)
                if action_name in self.tools:
                    observation = self.tools[action_name].execute(action_input)
                    tool_executed = True
                else:
                    observation = f"Tool '{action_name}' not found. Choose from: {', '.join(self.tools)}"
                self.memory.add(Message(role=Role.USER, content=f"Observation: {observation}"))

                # A final answer generated before the observation is not verified.
                if "Final Answer:" in llm_response:
                    self.memory.add(Message(
                        role=Role.USER,
                        content="Use the tool observation above to provide the verified final answer.",
                    ))
                continue

            if "Final Answer:" in llm_response:
                if not self.tools or tool_executed:
                    return llm_response.rsplit("Final Answer:", 1)[1].strip()

                logger.warning("Rejected final answer; no tool has been executed.")
                self.memory.add(Message(
                    role=Role.USER,
                    content=(
                        "System: You answered directly without executing an available tool. "
                        "Please execute the required tool first."
                    ),
                ))
                continue

            self.memory.add(Message(
                role=Role.USER,
                content="Observation: Your output must include Thought, Action, and Action Input.",
            ))

        return "Max iterations reached without finding a final answer."


if __name__ == "__main__":
    agent = ReActAgent([
        Tool("Calculator", "Evaluates a basic arithmetic expression, e.g. 12 * (3 + 4).", calculator)
    ])
    print(agent.run("What is (42 * 17) + 8? Use the Calculator tool."))
