"""
app.py
Streamlit chat UI for the agent: type a task, watch it think, call
tools, and produce a final answer, with the full trace visible in an
expander for transparency.
Run with: streamlit run app.py
"""

import streamlit as st
from dotenv import load_dotenv

from src.agent import Agent

load_dotenv()

st.set_page_config(page_title="Tool-Using Agent Demo", page_icon="🤖", layout="wide")
st.title("🤖 Tool-Using Agent")
st.caption("Give it a multi-step task. It can use a calculator, web search, and a Python sandbox.")

if "agent" not in st.session_state:
    st.session_state.agent = Agent()

task = st.text_area(
    "Task",
    placeholder="e.g. Compute compound interest on $1000 at 5% for 10 years, then explain the result.",
    height=100,
)
run = st.button("Run agent", type="primary", disabled=not task)

if run and task:
    with st.spinner("Agent working..."):
        result = st.session_state.agent.run(task)

    st.subheader("Final answer")
    st.write(result["answer"])

    col1, col2, col3 = st.columns(3)
    col1.metric("Steps taken", result["steps_taken"])
    col2.metric("Tool calls", result["tool_calls"])
    col3.metric("Failed tool calls", result["failed_tool_calls"])

    with st.expander("Full reasoning trace"):
        for entry in result["trace"]:
            st.markdown(f"**Step {entry['step']}**")
            if entry.get("thought"):
                st.write(entry["thought"])
            if entry.get("tool"):
                st.code(f"tool: {entry['tool']}\ninput: {entry['input']}\noutput: {entry['output']}")
            st.divider()
