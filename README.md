# Memory and State Management

An agent-oriented project demonstrating short-term memory, semantic memory, episodic memory, ReAct tool use, planning, evaluation, and hierarchical worker delegation.

## Project layout

```text
Memory/
├── app.py                         # Streamlit dashboard entry point
├── src/memory_manager/            # Reusable application package
├── examples/                      # Runnable demonstrations
├── .env.example                   # Provider configuration template
├── pyproject.toml                 # Package and dependency metadata
└── README.md
```

The detailed concept run is written to `docs/concept_execution.log`.

Set the provider and credentials in `.env`. For Ollama Cloud:

```env
LLM_PROVIDER=ollama
OLLAMA_API_KEY=your_ollama_cloud_api_key
OLLAMA_MODEL=gpt-oss:120b
```

## Run

```powershell
streamlit run app.py
python -m examples.basic_react
python -m examples.episodic_demo
python -m examples.planning_demo
python -m examples.hierarchical_demo
python -m unittest discover -s tests
python scripts/run_all_concepts.py
```

ChromaDB memory is currently in-memory
