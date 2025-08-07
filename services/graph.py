import json
from typing import Dict, Any, List, TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages  # Import add_messages for state management

import logging

logger = logging.getLogger(__name__)


# Define the State. Using Annotated to ensure messages are appended.
# This is a common and recommended pattern in LangGraph.
class State(TypedDict):
    messages: Annotated[List[Any], add_messages]  # Use Any for now, but BaseMessage is better


def route_next_step(state: State):
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        print(f"Routing: LLM requested tool calls. Going to 'invoke_tools'. Tool calls: {last_message.tool_calls}")
        return "invoke_tools"
    elif isinstance(last_message, ToolMessage):
        # After a tool is invoked, we want to go back to the LLM
        # to interpret the tool output and potentially generate a final answer
        print(
            "Routing: Tool invoked. Going back to 'call_tools_llm' to interpret output or to 'generate_final_answer'.")
        # This is a crucial decision point.
        # You might want to go back to call_tools_llm (your orchestrator)
        # to decide what to do next based on tool results, or directly to
        # a final answer generation if the tool call was the last step.
        # For now, let's assume the orchestrator should always see tool results
        # and decide.
        return "call_tools_llm"  # Let the orchestrator LLM decide the next step after tool output
    else:
        print("Routing: No tool calls. Ending graph execution.")
        return END


class GraphService:
    def __init__(self, Tools=None):
        self.graph = None
        self.tools = Tools
        AGENT_MODEL = "gpt-4o"
        self._tools_llm = ChatOpenAI(
            model=AGENT_MODEL,
            temperature=0,
        ).bind_tools(Tools)

        # New LLM for generating final, human-readable responses and summaries
        # It's good to separate this from the tool-calling LLM if their roles are distinct.
        self._response_llm = ChatOpenAI(
            model=AGENT_MODEL,
            temperature=0.7,  # A bit higher temperature for more creative synthesis
        )

    # NEW FUNCTION: To extract knowledge base file names
    def _extract_knowledge_base_files(self, messages: List[Any]) -> List[str]:
        """
        Extracts file names from search_knowledge_base tool outputs in the messages history.
        """
        file_names = []
        for msg in messages:
            if isinstance(msg, ToolMessage) and msg.name == "search_knowledge_base":
                try:
                    # The content of ToolMessage for search_knowledge_base is a JSON string
                    tool_output = json.loads(msg.content)
                    if "files" in tool_output and isinstance(tool_output["files"], list):
                        file_names.extend(tool_output["files"])
                except json.JSONDecodeError:
                    logger.warning(f"Could not decode JSON from ToolMessage content: {msg.content}")
                except Exception as e:
                    logger.error(f"Error extracting files from search_knowledge_base tool output: {e}")
        # Return unique file names
        return list(set(file_names))

    # Modified route_next_step

    async def call_tools_llm(self, state: State):
        try:
            messages = state["messages"]

            system_prompt = """

            You are a Senior Strategy Supervisor managing a team of AI consulting specialists at WPP. Your role is to intelligently select and orchestrate the most relevant agents based on the user's specific query.
            
            Available Specialist Agents:
            
            -  strategy_orchestrator : Complex strategic analysis, business model innovation, strategic planning, competitive positioning
            -  market_research_agent : Market analysis, competitive intelligence, industry trends, customer insights, market sizing
            -  technical_architect_agent : System design, technology feasibility, digital transformation, implementation planning
            -  financial_analyst_agent : ROI analysis, cost planning, financial projections, budgeting, financial performance optimization
            -  risk_assessment_agent : Risk identification, compliance, mitigation strategies, regulatory analysis
            -  data_scientist_agent : Statistical analysis, predictive modeling, quantitative insights, data-driven recommendations
            -  search_knowledge_base : Internal company documentation, policies, procedures, and institutional expertise
            -  search_google : External market intelligence, industry reports, competitor information, best practices
            
            Agent Selection Logic:
            
            Primary Keywords → Lead Agent(s):
            - Financial terms  (revenue, profit, costs, budget, ROI, cash flow, financial position) → `financial_analyst_agent` + relevant supporting agents
            - Market terms  (competition, customers, market share, industry trends, positioning) → `market_research_agent` + relevant supporting agents  
            - Technology terms  (digital transformation, systems, platform, automation, tech stack) → `technical_architect_agent` + relevant supporting agents
            - Risk terms  (compliance, regulations, threats, vulnerabilities, governance) → `risk_assessment_agent` + relevant supporting agents
            - Data terms  (analytics, insights, modeling, predictions, metrics) → `data_scientist_agent` + relevant supporting agents
            - Strategic terms  (vision, strategy, transformation, growth, innovation) → `strategy_orchestrator` + domain-specific agents
            
            ### Agent Selection Principles:
            - Identify the  primary domain  from the query to select the lead agent
            - Consider  secondary aspects  that may require supporting agents
            - Use multiple agents when the query spans multiple domains or requires cross-functional analysis
            - Always ensure the most relevant domain expert leads the analysis
            
             Decision Workflow:
            
            1. Query Analysis : Identify primary domain and secondary considerations from the user's question
            2. Agent Selection : Choose lead agent based on primary domain + supporting agents for secondary aspects
            3. Information Gathering : Use search tools if external data is needed
            4. Execution : Call selected agents in logical sequence (lead agent last for synthesis)
            5. Synthesis : Deliver comprehensive executive-level response
            
             Response Structure:
            -  Executive Summary : Key findings and recommendations
            -  Domain Analysis : Insights from each called agent
            -  Risk Considerations : Potential challenges and mitigation
            -  Implementation Roadmap : Prioritized action steps
            -  Success Metrics : KPIs to track progress
            
             Example Query Routing:
            
             "How can I improve company's financial position?" 
            → Lead: `financial_analyst_agent` 
            → Supporting: `market_research_agent` (revenue opportunities), `strategy_orchestrator` (strategic recommendations)
            
             "Should we enter the AI market?" 
            → Lead: `market_research_agent`
            → Supporting: `technical_architect_agent` (feasibility), `financial_analyst_agent` (investment), `risk_assessment_agent` (market risks)
            
            Always prioritize the most relevant domain expert as the lead agent, then add supporting agents that provide critical complementary perspectives.
            """

            # The current messages already include the original HumanMessage,
            # AIMessage with tool calls (if any), and ToolMessage with tool outputs (if any).
            # We need to ensure the system prompt is at the beginning for each LLM call.
            messages_for_llm = [SystemMessage(content=system_prompt)] + messages

            print(f"Calling _tools_llm with messages: {messages_for_llm}")
            message = await self._tools_llm.ainvoke(messages_for_llm)
            print(f"Response from _tools_llm: {message}")

            # The return value for the node function should be a dictionary
            # that updates the state. Using 'messages' key for appending.
            return {"messages": [message]}

        except Exception as e:
            print("error in call tools llm", e)
            logger.error(f"Error in call_tools_llm: {str(e)}")
            raise e

    def create_graph(self):
        """Creates and compiles the workflow graph."""
        try:
            # Create graph with state
            builder = StateGraph(State)

            builder.set_entry_point("call_tools_llm")

            builder.add_node("call_tools_llm", self.call_tools_llm)
            builder.add_node("invoke_tools", ToolNode(self.tools))

            # Define edges
            builder.add_edge(START, "call_tools_llm")  # Start always goes to call_tools_llm

            # Conditional edges from call_tools_llm
            builder.add_conditional_edges("call_tools_llm", route_next_step)
            # The route_next_step function will handle transitions to 'invoke_tools' or END.

            # After invoking tools, always go back to the call_tools_llm to let it interpret
            # the tool's output and decide what to do next (e.g., call another tool, or end)
            builder.add_edge("invoke_tools", "call_tools_llm")

            # Compile the graph
            self.graph = builder.compile()
            print("Graph compiled successfully.")

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
            # Max iterations to prevent infinite loops in complex graphs
            final_state = await self.graph.ainvoke(state, {"recursion_limit": 50})
            logger.info(f"Graph processing complete")

            # Extract final answer
            messages = final_state["messages"]
            print("Final state messages:", messages)

            # The final answer should be the last AIMessage generated by your LLM
            # after it has synthesized all information.
            final_answer_message = next(
                (msg for msg in reversed(messages) if isinstance(msg, AIMessage) and not msg.tool_calls), None)
            final_answer = final_answer_message.content if final_answer_message else "No comprehensive response generated."

            # Prepare the base response dictionary
            response_data = {
                'final_answer': final_answer,
                'used_tools': [] # Initialize as empty list
            }

            # Populate used_tools
            for msg in messages:
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        if tool_call['name'] not in response_data['used_tools']:
                            response_data['used_tools'].append(tool_call['name'])
                elif isinstance(msg, ToolMessage):
                    # Make sure to add the name of the tool that was actually executed if it's not already there
                    if msg.name not in response_data['used_tools']:
                        response_data['used_tools'].append(msg.name)

            # --- Conditional Logic for knowledge_base_files_used ---
            knowledge_base_files = self._extract_knowledge_base_files(messages)
            if knowledge_base_files:  # Check if the list of files is not empty
                response_data['knowledge_base_files_used'] = knowledge_base_files
            # --- End Conditional Logic ---

            return response_data

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return f"Error processing request: {str(e)}"
