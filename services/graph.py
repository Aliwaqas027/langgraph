from typing import Literal
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.messages import HumanMessage
from langgraph.types import Command

import logging

logger = logging.getLogger(__name__)


class State(MessagesState):
    """State object for the graph."""
    next: str


class GraphService:
    def __init__(self, supervisor_service, research_agent, code_agent, frontend_agent, designer_agent):
        self.supervisor = supervisor_service
        self.research_agent = research_agent
        self.code_agent = code_agent
        self.frontend_agent = frontend_agent
        self.designer_agent = designer_agent
        self.graph = None

    def create_research_node(self):
        """Creates the researcher node."""

        def research_node(state: State) -> Command[Literal["supervisor"]]:
            logger.info("Executing research node")
            try:
                result = self.research_agent.invoke(state)
                logger.info(f"Research result: {result['messages'][-1].content[:100]}...")
                return Command(
                    update={
                        "messages": [
                            HumanMessage(content=result["messages"][-1].content, name="researcher")
                        ]
                    },
                    goto="supervisor"
                )
            except Exception as e:
                logger.error(f"Error in research node: {str(e)}")
                raise

        return research_node

    def create_code_node(self):
        """Creates the backend node."""

        def code_node(state: State) -> Command[Literal["supervisor"]]:
            logger.info("Executing backend node")
            try:
                result = self.code_agent.invoke(state)
                logger.info(f"Backend result: {result['messages'][-1].content[:100]}...")
                return Command(
                    update={
                        "messages": [
                            HumanMessage(content=result["messages"][-1].content, name="backend")
                        ]
                    },
                    goto="supervisor"
                )
            except Exception as e:
                logger.error(f"Error in code node: {str(e)}")
                raise

        return code_node

    def create_fe_node(self):

        def fe_node(state: State) -> Command[Literal["supervisor"]]:
            logger.info("Executing frontend node")
            try:
                result = self.frontend_agent.invoke(state)
                logger.info(f"Frontend result: {result['messages'][-1].content[:100]}...")
                return Command(
                    update={
                        "messages": [
                            HumanMessage(content=result["messages"][-1].content, name="frontend")
                        ]
                    },
                    goto="supervisor"
                )
            except Exception as e:
                logger.error(f"Error in fe_node node: {str(e)}")
                raise

        return fe_node

    def create_design_node(self):

        def design_node(state: State) -> Command[Literal["supervisor"]]:
            logger.info("Executing designer node")
            try:
                result = self.designer_agent.invoke(state)
                logger.info(f"Designer result: {result['messages'][-1].content[:100]}...")
                return Command(
                    update={
                        "messages": [
                            HumanMessage(content=result["messages"][-1].content, name="designer")
                        ]
                    },
                    goto="supervisor"
                )
            except Exception as e:
                logger.error(f"Error in design node: {str(e)}")
                raise

        return design_node

    def create_graph(self):
        """Creates and compiles the workflow graph."""
        try:
            # Create graph with state
            builder = StateGraph(State)

            # Add nodes
            builder.add_node("supervisor", self.supervisor.create_supervisor_node())
            builder.add_node("researcher", self.create_research_node())
            builder.add_node("backend", self.create_code_node())
            builder.add_node("designer", self.create_design_node())
            builder.add_node("frontend", self.create_fe_node())

            # Add edges
            builder.add_edge(START, "supervisor")
            builder.add_edge("researcher", "supervisor")
            builder.add_edge("backend", "supervisor")
            builder.add_edge("designer", "supervisor")
            builder.add_edge("frontend", "supervisor")

            # Compile graph
            self.graph = builder.compile()
            logger.info("Graph created successfully")

        except Exception as e:
            logger.error(f"Error creating graph: {str(e)}")
            raise

    def process_query(self, query: str) -> str:
        """Process a query through the graph."""
        try:
            if not self.graph:
                logger.info("Creating new graph...")
                self.create_graph()

            # Initialize state with query
            state = {"messages": [HumanMessage(content=query)]}
            logger.info(f"Initial state created with query: {query}")

            # Process through graph
            final_state = self.graph.invoke(state)
            logger.info("Graph processing complete")

            # Extract final answer
            messages = final_state["messages"]
            responses = []

            for msg in messages:
                if hasattr(msg, 'name') and msg.name in ['researcher', 'backend', 'frontend', 'designer']:
                    responses.append(f"{msg.name.capitalize()}: {msg.content}")

            return "\n\n".join(responses) if responses else "No response generated"

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return f"Error processing request: {str(e)}"
