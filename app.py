# app.py
import time
import json
import streamlit as st
import pandas as pd
from datetime import datetime

# ==============================================================================
# PAGE CONFIGURATION & STYLING
# ==============================================================================
st.set_page_config(
    page_title="Agent Execution & Memory Inspector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .thought-card {
        background-color: #1E222A;
        border-left: 4px solid #61AFEF;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 10px;
    }
    .action-card {
        background-color: #242B28;
        border-left: 4px solid #98C379;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 10px;
    }
    .observation-card {
        background-color: #2A2421;
        border-left: 4px solid #E5C07B;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 10px;
    }
    .badge-vector {
        background-color: #8C52FF22;
        color: #B28DFF;
        border: 1px solid #8C52FF55;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# SESSION STATE INITIALIZATION
# ==============================================================================
if "execution_logs" not in st.session_state:
    st.session_state.execution_logs = []
if "vector_queries" not in st.session_state:
    st.session_state.vector_queries = []
if "is_running" not in st.session_state:
    st.session_state.is_running = False

# ==============================================================================
# MOCK AGENT ENGINE WITH EVENT CALLBACKS
# ==============================================================================
class AgentExecutionEngine:
    def __init__(self, trace_callback, memory_callback):
        self.trace_cb = trace_callback
        self.mem_cb = memory_callback

    def run_agent(self, prompt: str):
        # Step 1: Initial Thought
        self.trace_cb("Thought", f"Analyzing prompt: '{prompt}'. Need to check knowledge base for local city statistics.", step=1)
        time.sleep(0.8)

        # Vector Query 1
        self.mem_cb(
            query="Coimbatore infrastructure development budget 2026",
            results=[
                {"id": "doc_101", "score": 0.92, "text": "Coimbatore 2026 budget allocates 4.5 billion INR across road, transit, and water projects.", "metadata": {"source": "govt_report_2026.pdf"}},
                {"id": "doc_102", "score": 0.81, "text": "Smart water management received 25% of municipal development allocations.", "metadata": {"source": "water_dept_summary.txt"}},
                {"id": "doc_103", "score": 0.65, "text": "Public transit improvements include electric bus fleet upgrades in TN.", "metadata": {"source": "transit_board.pdf"}}
            ]
        )
        time.sleep(1.0)

        # Step 1: Observation
        self.trace_cb("Observation", "Retrieved 3 vector matches. Highest relevance (0.92): Total budget is 4.5B INR.", step=1)
        time.sleep(0.6)

        # Step 2: Thought & Action
        self.trace_cb("Thought", "Now calculating the specific allocation for public transit (35% of total).", step=2)
        time.sleep(0.6)
        
        self.trace_cb("Action", "Calculator.eval(expr='4500000000 * 0.35')", tool="Calculator", step=2)
        time.sleep(1.0)

        # Step 2: Observation
        self.trace_cb("Observation", "Tool Output: 1575000000.0 (1.575 Billion INR)", step=2)
        time.sleep(0.8)

        # Step 3: Thought & Final Output
        self.trace_cb("Thought", "Synthesis complete. Formulating clear breakdown response.", step=3)
        time.sleep(0.5)

        return (
            "### Summary Response\n\n"
            "The total infrastructure budget for Coimbatore in 2026 is **₹4.5 Billion (450 Crore INR)**.\n\n"
            "**Key Allocations:**\n"
            "- **Road Development (40%):** ₹1.80 Billion\n"
            "- **Public Transit (35%):** ₹1.575 Billion\n"
            "- **Smart Water Management (25%):** ₹1.125 Billion"
        )

# ==============================================================================
# UI SIDEBAR CONTROLS
# ==============================================================================
with st.sidebar:
    st.title("⚙️ Execution Settings")
    st.markdown("---")
    
    agent_model = st.selectbox("LLM Backbone", ["Gemini 2.5 Flash", "Llama 3.3 70B", "Claude 3.5 Sonnet"])
    vector_metric = st.selectbox("Distance Metric", ["Cosine Similarity", "Dot Product", "Euclidean (L2)"])
    top_k = st.slider("Vector Top-K", min_value=1, max_value=10, value=3)
    
    st.markdown("---")
    if st.button("🧹 Clear Execution Traces", use_container_width=True):
        st.session_state.execution_logs = []
        st.session_state.vector_queries = []
        st.rerun()

# ==============================================================================
# MAIN APPLICATION INTERFACE
# ==============================================================================
st.title("🤖 ReAct Agent & Memory Inspector")
st.caption("Real-time visual trace of Thought-Action-Observation loops and Vector Similarity retrievals.")

user_prompt = st.text_input(
    "User Prompt", 
    value="What is the public transit budget allocation for Coimbatore in 2026?",
    placeholder="Enter prompt..."
)

col_run, col_status = st.columns([1, 4])
with col_run:
    start_btn = st.button("🚀 Run Agent Loop", type="primary", use_container_width=True)

# Main Dashboard Layout Tabs
tab_trace, tab_vector, tab_raw = st.tabs([
    "🔄 ReAct Loop Trace", 
    "🎯 Vector Memory Retrievals", 
    "📊 Raw JSON & Metrics"
])

# ==============================================================================
# REAL-TIME EXECUTION LOGIC
# ==============================================================================
if start_btn and user_prompt:
    st.session_state.execution_logs = []
    st.session_state.vector_queries = []
    st.session_state.is_running = True

    # Containers for live updating UI
    with tab_trace:
        status_box = st.status("Agent processing request...", expanded=True)
        live_trace_container = st.container()

    # Define Event Handlers
    def handle_trace_event(event_type: str, content: str, tool: str = None, step: int = 1):
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        log_entry = {
            "timestamp": timestamp,
            "step": step,
            "type": event_type,
            "content": content,
            "tool": tool
        }
        st.session_state.execution_logs.append(log_entry)
        
        # Render live update inside st.status
        with status_box:
            if event_type == "Thought":
                st.markdown(f"**Step {step} - Thought:** {content}")
            elif event_type == "Action":
                st.code(f"Tool Call [{tool}]: {content}", language="python")
            elif event_type == "Observation":
                st.info(f"**Observation:** {content}")

    def handle_memory_event(query: str, results: list):
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        st.session_state.vector_queries.append({
            "timestamp": timestamp,
            "query": query,
            "results": results
        })

    # Instantiate and Run
    engine = AgentExecutionEngine(handle_trace_event, handle_memory_event)
    final_response = engine.run_agent(user_prompt)
    
    status_box.update(label="Execution Complete!", state="complete", expanded=False)
    st.session_state.is_running = False

    # Display final synthesized output
    with tab_trace:
        st.markdown("### 🏁 Final Agent Output")
        st.success(final_response)

# ==============================================================================
# TAB 1: REACT LOOP TRACE VIEW
# ==============================================================================
with tab_trace:
    if not start_btn and st.session_state.execution_logs:
        st.subheader("Execution History")
        
        current_step = None
        for log in st.session_state.execution_logs:
            if log["step"] != current_step:
                current_step = log["step"]
                st.markdown(f"#### 📍 Iteration Step {current_step}")
            
            ts = log['timestamp']
            if log["type"] == "Thought":
                st.markdown(
                    f"<div class='thought-card'><b>[{ts}] 🤔 THOUGHT</b><br>{log['content']}</div>", 
                    unsafe_allow_html=True
                )
            elif log["type"] == "Action":
                st.markdown(
                    f"<div class='action-card'><b>[{ts}] ⚡ ACTION ({log['tool']})</b><br><code>{log['content']}</code></div>", 
                    unsafe_allow_html=True
                )
            elif log["type"] == "Observation":
                st.markdown(
                    f"<div class='observation-card'><b>[{ts}] 👁️ OBSERVATION</b><br>{log['content']}</div>", 
                    unsafe_allow_html=True
                )

# ==============================================================================
# TAB 2: VECTOR MEMORY INSPECTOR
# ==============================================================================
with tab_vector:
    st.subheader("Semantic Vector Store Queries")
    
    if not st.session_state.vector_queries:
        st.info("No vector queries captured yet. Run the agent to inspect embeddings search.")
    else:
        for idx, query_evt in enumerate(st.session_state.vector_queries, 1):
            with st.expander(f"🔍 Query #{idx}: \"{query_evt['query']}\" ({query_evt['timestamp']})", expanded=True):
                st.caption(f"Search Query: **{query_evt['query']}**")
                
                # Render matches as dataframe and expanded cards
                results = query_evt["results"]
                df_results = pd.DataFrame(results)
                
                # Progress bar display for scores
                st.markdown("##### Top Matching Chunks")
                for r in results:
                    c1, c2 = st.columns([1, 4])
                    with c1:
                        st.metric("Similarity", f"{r['score']:.2f}")
                        st.progress(r['score'])
                    with c2:
                        st.markdown(f"**Chunk ID:** `{r['id']}` <span class='badge-vector'>{r['metadata']['source']}</span>", unsafe_allow_html=True)
                        st.write(r['text'])
                    st.divider()

# ==============================================================================
# TAB 3: RAW METRICS & EVENT STREAM JSON
# ==============================================================================
with tab_raw:
    st.subheader("Agent Execution Analytics")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Loop Steps", len(set(x['step'] for x in st.session_state.execution_logs)))
    col2.metric("Total Events Emitted", len(st.session_state.execution_logs))
    col3.metric("Vector Queries Issued", len(st.session_state.vector_queries))
    
    st.markdown("### Raw Event Stream (JSON)")
    st.json({
        "trace_events": st.session_state.execution_logs,
        "vector_queries": st.session_state.vector_queries
    })