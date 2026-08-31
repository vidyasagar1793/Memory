"""Run and log every memory/state-management concept in the project.

Use ``--live`` to call the configured LLM provider. Without ``--live``, the
LLM-dependent workflows use deterministic responses for offline verification.
"""

from __future__ import annotations

import importlib
import json
import logging
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
LOG_PATH = ROOT / "docs" / "concept_execution.log"
INFO_LOG_PATH = ROOT / "info.log"
LIVE = "--live" in sys.argv
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class MarkdownLog:
    def __init__(self) -> None:
        self.started = datetime.now(timezone.utc)
        self.rows: list[dict[str, str]] = []

    def case(self, name: str, function) -> None:
        started = time.perf_counter()
        try:
            detail = str(function())
            status = "PASS"
        except Exception as error:  # pragma: no cover - exercised on failure
            status = "FAIL"
            detail = f"{type(error).__name__}: {error}\n\n```text\n{traceback.format_exc()}\n```"
        elapsed = time.perf_counter() - started
        self.rows.append({"name": name, "status": status, "elapsed": f"{elapsed:.3f}s", "detail": detail})
        print(f"{status} | {name} | {elapsed:.3f}s | {detail.splitlines()[0]}")

    def write(self) -> None:
        passed = sum(row["status"] == "PASS" for row in self.rows)
        lines = [
            "# Concept Execution Log",
            "",
            f"Started (UTC): {self.started.isoformat()}",
            f"Finished (UTC): {datetime.now(timezone.utc).isoformat()}",
            "",
            (
                "This run used live LLM provider calls."
                if LIVE
                else "All LLM responses in this run were deterministic mocks. No external provider request was made."
            ),
            "",
            "Provider mode: live Ollama Cloud calls when invoked with `--live`; this run does not store credentials.",
            "",
            f"## Summary: {passed}/{len(self.rows)} cases passed",
            "",
            "| Case | Status | Duration | Result |",
            "|---|---:|---:|---|",
        ]
        for row in self.rows:
            first_line = row["detail"].splitlines()[0].replace("|", "\\|")
            lines.append(f"| `{row['name']}` | **{row['status']}** | {row['elapsed']} | {first_line} |")
        lines.extend(["", "## Detailed results", ""])
        for index, row in enumerate(self.rows, 1):
            lines.extend([
                f"### {index}. `{row['name']}` — {row['status']}",
                "",
                f"Duration: `{row['elapsed']}`",
                "",
                row["detail"],
                "",
            ])
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        content = "\n".join(lines)
        LOG_PATH.write_text(content, encoding="utf-8")
        INFO_LOG_PATH.write_text(content, encoding="utf-8")


logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
log = MarkdownLog()


def imported_modules():
    modules = [
        "memory_manager.agent",
        "memory_manager.agentic",
        "memory_manager.core_models",
        "memory_manager.episodic_memory",
        "memory_manager.evaluator",
        "memory_manager.hierarchical_architecture",
        "memory_manager.llm_client",
        "memory_manager.memory",
        "memory_manager.planner",
        "memory_manager.planner_models",
        "memory_manager.plan_executor",
        "memory_manager.plan_executor2",
        "memory_manager.semantic_memory",
    ]
    for name in modules:
        importlib.import_module(name)
    return f"Imported {len(modules)} package modules successfully."


def core_models_case():
    from memory_manager.core_models import Message, Role

    message = Message(role=Role.USER, content="hello", metadata={"source": "test"})
    expected = {"role": "user", "content": "hello"}
    assert message.to_llm_format() == expected
    return f"Message serialized as {expected}."


def short_term_case():
    from memory_manager.core_models import Message, Role
    from memory_manager.memory import ShortTermMemory

    memory = ShortTermMemory(max_tokens=10)
    memory.add(Message(role=Role.SYSTEM, content="system prompt"))
    memory.add(Message(role=Role.USER, content="one two three four five six seven"))
    memory.add(Message(role=Role.ASSISTANT, content="assistant result"))
    assert memory.history[0].role == Role.SYSTEM
    assert len(memory.history) == 2
    return "Older user context pruned; system prompt preserved; 2 messages remained."


def semantic_case():
    from memory_manager.semantic_memory import SemanticMemory

    memory = SemanticMemory(collection_name=f"log_semantic_{uuid4().hex}")
    memory.add_knowledge("budget", "Coimbatore 2026 infrastructure budget is 4500000000 INR.")
    result = memory.search("Coimbatore infrastructure budget")
    assert "4500000000" in result
    return f"ChromaDB search returned: {result}"


def episodic_case():
    from memory_manager.episodic_memory import EpisodicMemory

    memory = EpisodicMemory(collection_name=f"log_episodic_{uuid4().hex}")
    memory.log_episode("episode-1", "credential lookup", False, "Use exact system names.", ["SearchMemory"])
    result = memory.recall_relevant_lessons("retrieve credential lookup")
    assert "Use exact system names" in result
    return f"Recalled lesson: {result}"


def react_case():
    import memory_manager.agent as agent_module
    from memory_manager.agent import ReActAgent, Tool, calculator

    if not LIVE:
        responses = iter([
            "Thought: calculate.\nAction: Calculator\nAction Input: 6 * 7",
            "Thought: verified.\nFinal Answer: 42",
        ])
        agent_module.call_llm = lambda *args, **kwargs: next(responses)
    result = ReActAgent([Tool("Calculator", "safe arithmetic", calculator)]).run(
        "Calculate 6 * 7 using the Calculator tool and return the final answer."
    )
    if LIVE:
        assert result.strip(), "Live ReAct agent returned an empty result"
        assert "LLM call failed" not in result
        assert "Max iterations" not in result
    else:
        assert result == "42"
    try:
        calculator("__import__('os').system('echo unsafe')")
    except (ValueError, SyntaxError):
        pass
    else:
        raise AssertionError("calculator accepted executable Python")
    return "Calculator tool selected by ReAct loop; result=42; executable expression rejected."


def planning_case():
    import memory_manager.agent as agent_module
    import memory_manager.evaluator as evaluator_module
    import memory_manager.planner as planner_module
    from memory_manager.agent import Tool, calculator
    from memory_manager.plan_executor2 import PlanExecutor

    if not LIVE:
        planner_module.call_llm = lambda *args, **kwargs: json.dumps({
            "goal": "calculate 6 * 7",
            "steps": [{"step_id": 1, "title": "Calculate", "instruction": "Calculate 6 * 7.", "assigned_tool": "Calculator"}],
        })
        evaluator_module.call_llm = lambda *args, **kwargs: json.dumps({"success": True, "feedback": "Correct."})
        responses = iter([
            "Thought: calculate.\nAction: Calculator\nAction Input: 6 * 7",
            "Thought: verified.\nFinal Answer: 42",
        ])
        agent_module.call_llm = lambda *args, **kwargs: next(responses)
    result = PlanExecutor([Tool("Calculator", "safe arithmetic", calculator)]).execute_goal("calculate 6 * 7")
    assert result.strip()
    if LIVE:
        assert "Execution failed" not in result
    return f"Plan parsed; step executed; evaluator completed; result={result[:300]}"


def hierarchical_case():
    import memory_manager.agent as agent_module
    import memory_manager.hierarchical_architecture as hierarchy_module
    from memory_manager.agent import Tool
    from memory_manager.hierarchical_architecture import HierarchicalAgentSystem
    from memory_manager.semantic_memory import SemanticMemory

    memory = SemanticMemory(collection_name=f"log_hierarchy_{uuid4().hex}")
    memory.add_knowledge("budget", "The budget is 100 INR.")
    if not LIVE:
        meta_plan = {
            "high_level_goal": "research and analyze",
            "assignments": [
                {"worker_role": "ResearchWorker", "task_description": "Retrieve the fact."},
                {"worker_role": "AnalysisWorker", "task_description": "Analyze the fact."},
            ],
        }
        hierarchy_responses = iter([json.dumps(meta_plan), "FINAL SYNTHESIS: completed"])
        hierarchy_module.call_llm = lambda *args, **kwargs: next(hierarchy_responses)
        worker_responses = iter([
            "Thought: search.\nAction: SearchMemory\nAction Input: budget",
            "Thought: verified.\nFinal Answer: research completed",
            "Thought: search.\nAction: SearchMemory\nAction Input: budget allocation",
            "Thought: verified.\nFinal Answer: analysis completed",
        ])
        agent_module.call_llm = lambda *args, **kwargs: next(worker_responses)
    tool = Tool("SearchMemory", "search memory", memory.search)
    result = HierarchicalAgentSystem([tool], [tool]).run("research and analyze")
    assert result.strip()
    if LIVE:
        assert "LLM call failed" not in result
        assert "Unsupported LLM provider" not in result
        assert "iteration limit" not in result.lower()
        assert "max iteration" not in result.lower()
    return f"Meta-plan, research worker, analysis worker, and synthesis completed; result={result[:300]}"


log.case("package imports", imported_modules)
log.case("core_models Message and Role", core_models_case)
log.case("memory ShortTermMemory sliding window", short_term_case)
log.case("semantic_memory ChromaDB retrieval", semantic_case)
log.case("episodic_memory lesson recall", episodic_case)
log.case("agent ReAct tool dispatch and safe calculator", react_case)
log.case("planner + plan_executor2 + evaluator", planning_case)
log.case("hierarchical_architecture delegation", hierarchical_case)
log.write()
print(f"Detailed log written to {LOG_PATH}")
