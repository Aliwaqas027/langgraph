# services/graph.py
from typing import Annotated, TypedDict, List
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from agents.supervisor import supervisor_agent
from agents.llm_agent import llm_agent
from agents.google_agent import google_agent
from agents.pinecone_agent import knowledge_base_agent


class State(TypedDict):
    messages: Annotated[List, add_messages]
    agent_type: str


def build_graph():
    """
    Builds and returns the compiled graph for the multi-agent system.
    """
    # Create graph
    graph = StateGraph(State)

    # Add nodes
    graph.add_node("supervisor", supervisor_agent)
    graph.add_node("llm", llm_agent)
    graph.add_node("google", google_agent)
    graph.add_node("knowledge_base", knowledge_base_agent)

    # Add conditional edges from supervisor
    def route_agent(state):
        return state["agent_type"]

    graph.add_conditional_edges(
        "supervisor",
        route_agent,
        {
            "llm": "llm",
            "google": "google",
            "knowledge_base": "knowledge_base"
        }
    )

    # Add edges
    graph.add_edge(START, "supervisor")
    graph.add_edge("llm", END)
    graph.add_edge("google", END)
    graph.add_edge("knowledge_base", END)

    return graph.compile()
