# Memory and State Management

A learning project that builds up an agent stack one concept at a time: short-term
memory, semantic memory, parametric memory, episodic memory, prospective memory, ReAct tool use,
planning, evaluation, and
hierarchical worker delegation.

## Project layout

```text
Memory/
├── src/memory_manager/         # The library, grouped by concept
│   ├── core/                   #   Message models + multi-provider LLM client
│   │   ├── models.py           #     Message, Role
│   │   ├── llm_client.py       #     call_llm() across ollama/groq/gemini/hf
│   │   └── trace.py            #     Event bus the frontend subscribes to
│   ├── memory_stores/          #   The memory layers
│   │   ├── short_term.py       #     ShortTermMemory — token-pruned scratchpad
│   │   ├── semantic.py         #     SemanticMemory — facts in a vector store
│   │   ├── parametric.py       #     ParametricMemory — probe the model's beliefs, compare with evidence
│   │   ├── episodic.py         #     EpisodicMemory — lessons from past runs
│   │   └── prospective.py      #     ProspectiveMemory — reminders that fire on a cue
│   ├── agents/                 #   Agents and the tools they call
│   │   ├── tools.py            #     Tool wrapper + safe calculator
│   │   ├── react.py            #     ReActAgent — Thought/Action/Observation loop
│   │   └── hierarchical.py     #     HierarchicalAgentSystem — meta-planner + workers
│   └── planning/               #   Plan-and-execute pipeline
│       ├── models.py           #     TaskPlan, SubTask, StepStatus
│       ├── planner.py          #     Planner — goal -> plan
│       ├── evaluator.py        #     Evaluator — did this step actually succeed?
│       ├── executor.py         #     PlanExecutor — run each step
│       └── evaluating_executor.py  # PlanExecutor + evaluate-and-retry
├── webapp/                     # Web frontend that streams a run as it happens
│   ├── server.py               #   FastAPI + Server-Sent Events
│   ├── scenarios.py            #   One runnable scenario per concept
│   ├── offline.py              #   Scripted model, for running without an API key
│   └── static/                 #   The page itself (no build step)
├── examples/                   # Runnable demos, one per concept
├── scripts/run_all_concepts.py # Exercises every concept and writes a report
├── docs/                       # Project report and the latest execution log
├── .env.example                # Provider configuration template
└── pyproject.toml
```

Every public class is re-exported at the top level, so demos stay short:

```python
from memory_manager import ReActAgent, SemanticMemory, Tool
```

Import the submodule instead when you want to see where something lives
(`from memory_manager.planning.planner import Planner`).

## Setup

```bash
python -m venv .venv
.venv/bin/pip install -e .          # Windows: .venv\Scripts\pip install -e .
```

Set the provider and credentials in `.env` (copy `.env.example`). For Ollama Cloud:

```env
LLM_PROVIDER=ollama
OLLAMA_API_KEY=your_ollama_cloud_api_key
OLLAMA_MODEL=gpt-oss:120b
```

## Run the trace viewer

The fastest way to see how any of this works is the web frontend:

```bash
python -m webapp            # then open http://127.0.0.1:8000
```

Pick a concept, edit its inputs, press Run. The page streams the library's own
internals as they happen — the exact prompt sent to the model and its raw reply,
every tool call, every message entering or being pruned from the scratchpad, each
plan step changing status, and the evaluator's verdict on it. Side panels redraw
live: the short-term window with its token meter, the plan, and what is sitting in
the vector store.

Two modes, in the top right:

- **Live** — real calls using the credentials in `.env`.
- **Offline** — a scripted stand-in answers every model call, so the whole
  machinery (tool dispatch, retries, delegation) runs with no API key. Useful for
  reading the mechanics; it proves nothing about a real model's behaviour.

Nothing about this is a simulation of the library: the page runs the real classes
in `memory_manager` and draws the events they emit.

## Run the examples

Suggested reading order — each demo builds on the one above it:

```bash
python -m examples.memory_layers_demo   # the three memory layers, no LLM needed
python -m examples.basic_react          # ReAct agent + semantic memory search
python -m examples.parametric_demo      # weights alone vs. retrieval, then compare
python -m examples.episodic_demo        # inject lessons from past failures
python -m examples.prospective_demo     # arm a reminder, watch it fire on its cue
python -m examples.planning_demo        # plan -> execute -> evaluate -> retry
python -m examples.hierarchical_demo    # meta-planner delegating to workers
```

Verify every concept at once and write `docs/concept_execution.log`:

```bash
python scripts/run_all_concepts.py   # real calls to the configured provider
```

ChromaDB memory is currently in-memory, so semantic and episodic stores start empty
on every run.
