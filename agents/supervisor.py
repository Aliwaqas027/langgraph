from typing import Dict
from langchain_core.messages import HumanMessage
from langchain_openai import OpenAI

# Initialize LLM
llm = OpenAI()


def supervisor_agent(state: Dict):
    """
    Determines which agent should handle the query.
    Returns: Dict with agent_type key
    """
    messages = state["messages"]
    user_input = messages[-1].content if messages else ""

    system_prompt = """You are a supervisor agent that determines which specialized agent should handle a query.
    Options are:
    1. 'google' - For current events, factual queries needing web search
    2. 'knowledge_base' - For company-specific or domain-specific information
    3. 'llm' - For general conversation, opinions, or creative tasks
    
    Respond only with one of these options: 'google', 'knowledge_base', or 'llm'.
    Note: no other even a single word only respond with 'google', 'knowledge_base', or 'llm'.
    
    Example:
    User: "What is the capital of France?"
    google
    
    User: "Can you tell me about our company's policy?"
    knowledge_base
    
    User: "Write me a poem about the ocean."
    llm
    """

    messages = [
        HumanMessage(content=system_prompt),
        HumanMessage(content=f"Please decide which agent should handle this query: {user_input}")
    ]

    response = llm.invoke(messages)
    print("response==>",response.strip().lower())
    return {"agent_type": response.strip().lower()}
