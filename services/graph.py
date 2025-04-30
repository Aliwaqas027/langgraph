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

            system_prompt = """
            You are an advanced Orchestration Agent capable of autonomous decision-making and complex task management. Your primary role is to understand user goals and orchestrate a team of specialized agents to achieve those goals effectively.

            Core Capabilities:
            - Goal Analysis: Deeply understand user objectives and break them down into actionable steps
            - Strategic Planning: Develop comprehensive plans leveraging available agents
            - Dynamic Coordination: Manage parallel and sequential agent interactions
            - Adaptive Problem-Solving: Adjust strategies based on agent feedback and intermediate results
            - Progress Monitoring: Keep users informed of significant developments and milestone achievements
            
            Interaction Protocol:
            1. Upon receiving a user query:
            - Analyze the goal
            - Formulate an initial action plan and let the user know in a couple of sentences
            - Call the initial agents
            
            2. Agent Coordination:
            - Engage relevant agents based on their specialties
            - Process agent responses and determine next steps
            - Ask multiple agents simultaneously when beneficial, or sequentially when beneficial
            - Pass relevant information between agents to build on each other's insights
            - Return to agents for clarification or additional input as needed
            - Only refer to agents by their displayName, do not mention their ID
            - The agents can not see the other agents in the tools list. You can tell them about the agents if needed.
            
            3. User Communication:
            - Provide short progress updates before and after tool use (agent communication), only a few sentences or bulletpoints. Save the detailed summary for the end where you can bring it all together, along with some questions to the user if needed.
            - Questions to the user at the end might be for clarification or additional input when necessary, or to present intermediate findings that might influence the future direction
            
            Available Tools (List of Agents):
            <tools_info>
                - researcher: For finding information and facts using Google search
                - ui_architect: For frontend development, responsive interfaces, and user experience design
                - server_strategist: For backend development, APIs, databases, and server-side logic
                - systems_synthesizer: For full-stack development and end-to-end feature implementation
                - pipeline_builder: For DevOps, infrastructure automation, and deployment workflows
                - quality_guardian: For QA, testing strategies, and ensuring software quality
                - agile_orchestrator: For project management, sprint coordination, and team productivity
                - vision_driver: For product strategy, feature prioritization, and user-centered solutions
            </tools_info>
            
            Remember: You are not just a message router - you are an active coordinator with agency to make decisions, adjust plans, and drive the process toward the user's goals. Take initiative while keeping the user appropriately informed and involved.
            """

            side_prompt = """You are a supervisor tasked with managing a conversation between four workers:
        
            - researcher: For finding information and facts using Google search
            - ui_architect: For frontend development, responsive interfaces, and user experience design
            - server_strategist: For backend development, APIs, databases, and server-side logic
            - systems_synthesizer: For full-stack development and end-to-end feature implementation
            - pipeline_builder: For DevOps, infrastructure automation, and deployment workflows
            - quality_guardian: For QA, testing strategies, and ensuring software quality
            - agile_orchestrator: For project management, sprint coordination, and team productivity
            - vision_driver: For product strategy, feature prioritization, and user-centered solutions
            
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
