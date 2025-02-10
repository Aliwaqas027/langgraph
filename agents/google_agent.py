# services/agents/google_agent.py
from typing import Dict
from langchain_core.messages import HumanMessage
from langchain_openai import OpenAI
from utils.tools import search_google


# Initialize LLM
llm = OpenAI()

def google_agent(state: Dict):
    """
    Agent that uses Google Search.
    Returns: Dict with messages key
    """
    messages = state["messages"]
    last_message = messages[-1].content

    # First search Google
    search_results = search_google(last_message)

    # Then have LLM process the results
    prompt = f"""Based on the search results below, please provide a comprehensive answer to the query: {last_message}
    
    Search Results:
    {search_results}"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return {"messages": [response]}