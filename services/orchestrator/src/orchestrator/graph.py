"""
The LangGraph state machine.

Topology:
    data → forecast → critic → report → drift_check
                                          ├─ drift_detected=True  → __end__ (Phase 4 will retrigger)
                                          └─ drift_detected=False → __end__

Why a graph and not a sequence of function calls?
  - The LangGraph runtime gives us free benefits: visualization, tracing
    integration with Langfuse, conditional edges, checkpointing/replay.
  - Adding the retrigger path or branching (e.g. "if forecast is in crash
    territory, escalate to a human review queue") is a one-edge change.
  - Each node is independently testable and replaceable.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from orchestrator.nodes import (
    critic_node,
    data_node,
    drift_node,
    forecast_node,
    report_node,
)
from orchestrator.state import MarketState


def build_graph() -> StateGraph:
    """Construct the compiled LangGraph executor.

    Build ONCE at lifespan startup, reuse across requests. Graph compilation
    is cheap but non-trivial; we don't repeat it per /run call.
    """
    graph = StateGraph(MarketState)

    graph.add_node("data", data_node.run)
    graph.add_node("forecast", forecast_node.run)
    graph.add_node("critic", critic_node.run)
    graph.add_node("report", report_node.run)
    graph.add_node("drift", drift_node.run)

    graph.set_entry_point("data")
    graph.add_edge("data", "forecast")
    graph.add_edge("forecast", "critic")
    graph.add_edge("critic", "report")
    graph.add_edge("report", "drift")

    # Conditional terminus. The mapping says "if should_retrigger returns
    # True, end the graph (Phase 4 will route here to a retrigger node);
    # if False, also end." Same destination for now, but the SHAPE is in
    # place so Phase 4 just changes the True branch.
    graph.add_conditional_edges(
        "drift",
        drift_node.should_retrigger,
        {True: END, False: END},
    )

    return graph.compile()
