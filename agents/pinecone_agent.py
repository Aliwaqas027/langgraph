from typing import Dict
from langchain_core.messages import HumanMessage
from utils.tools import search_knowledge_base
from langchain_openai import OpenAI

# Initialize LLM
llm = OpenAI()


def knowledge_base_agent(state: Dict):
    messages = state["messages"]
    last_message = messages[-1].content
    print("state knowledge_base_agent=>", state["messages"])

    # Search knowledge base
    kb_results = search_knowledge_base(last_message)

    # Have LLM process the results
    prompt = f"""Based on the knowledge base results below, please provide a comprehensive answer to the query: {last_message}
    
    Knowledge Base Results:
    {kb_results}"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return {"messages": [response]}
