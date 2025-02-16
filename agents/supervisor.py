from typing import Literal
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import END
from langgraph.types import Command
import logging

logger = logging.getLogger(__name__)


class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""
    next: Literal["researcher", "coder", "FINISH"]


class SupervisorService:
    def __init__(self, model="gpt-4o"):
        self.llm = ChatOpenAI(model=model, temperature=0)

    def create_supervisor_node(self):
        """Creates the supervisor node for routing."""
        system_prompt = """You are a supervisor tasked with managing a conversation between four workers:
        
        - researcher: For finding information and facts using Google search
        - backend: For backend related information and queries
        - frontend: For frontend related information and queries
        - designer: For design related information and queries
    
        
        Important: Check the conversation history - if an agent has already provided their part,
        move to the next needed agent or FINISH if all tasks are complete."""

        def supervisor_node(state):
            try:
                # Format messages for LLM
                messages = [
                               {"role": "system", "content": system_prompt},
                           ] + state["messages"]

                logger.info("Supervisor processing request")

                # Get routing decision
                response = self.llm.with_structured_output(Router).invoke(messages)
                goto = response["next"]

                logger.info(f"Supervisor decision: {goto}")

                if goto == "FINISH":
                    goto = END

                # Return command with next step
                return Command(
                    goto=goto,
                    update={"next": goto}
                )

            except Exception as e:
                logger.error(f"Error in supervisor node: {str(e)}")
                raise

        return supervisor_node
