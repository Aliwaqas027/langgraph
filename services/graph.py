from typing import Dict, Any, List

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import ToolNode

import logging

logger = logging.getLogger(__name__)


class State(MessagesState):
    """State object for the graph."""
    next: str


class GraphService:
    def __init__(self, Tools=None):
        self.graph = None
        self.tools = Tools
        AGENT_MODEL = "gpt-4o"
        self._tools_llm = ChatOpenAI(
            model=AGENT_MODEL,
            temperature=0,
        ).bind_tools(Tools)

    @staticmethod
    def route_next_step(state: State):
        result = state["messages"][-1]
        if len(result.tool_calls) != 0:
            print("result.tool_calls", result.tool_calls)
            return "invoke_tools"
        return END

    async def call_tools_llm(self, state: State):
        try:
            messages = state["messages"]

            system_prompt = """You are a supervisor tasked with managing a conversation between four workers:
        
            - researcher: For finding information and facts using Google search
            - backend: For backend related information and queries
            - frontend: For frontend related information and queries
            - designer: For design related information and queries
            - legal: For legal queries
        
            
            Important: Check the conversation history - if an agent has already provided their part,
            move to the next needed agent or FINISH if all tasks are complete."""

            messages_with_system = [SystemMessage(content=system_prompt)] + messages

            message = await self._tools_llm.ainvoke(messages_with_system)

            return {"messages": [message]}

        except Exception as e:
            print("error in call tools llm", e)
            raise e

    def create_graph(self):
        """Creates and compiles the workflow graph."""
        try:
            # Create graph with state
            builder = StateGraph(State)

            builder.add_node("call_tools_llm", self.call_tools_llm)
            builder.add_node("invoke_tools", ToolNode(self.tools))

            builder.set_entry_point("call_tools_llm")

            builder.add_conditional_edges("call_tools_llm", self.route_next_step)

            # Compile the graph with tracing disabled
            self.graph = builder.compile()

        except Exception as e:
            logger.error(f"Error creating graph: {str(e)}")
            raise

    async def process_query(self, query: str) -> dict[str, str | list[Any] | Any] | str:
        """Process a query through the graph."""
        try:
            if not self.graph:
                logger.info("Creating new graph...")
                self.create_graph()

            # Initialize state with query
            state = {"messages": [HumanMessage(content=query)]}
            logger.info(f"Initial state created with query: {query}")

            # Process through graph
            final_state = await self.graph.ainvoke(state)
            logger.info(f"Graph processing complete")

            # Extract final answer
            messages = final_state["messages"]
            print("messages final state:", messages)

            messages = final_state['messages']

            final_answer = messages[-1].content if messages else "No response generated"

            used_tools = []
            for msg in messages:
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        if tool_call['name'] not in used_tools:
                            used_tools.append(tool_call['name'])

            return {
                'final_answer': final_answer,
                'used_tools': used_tools
            }

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return f"Error processing request: {str(e)}"
