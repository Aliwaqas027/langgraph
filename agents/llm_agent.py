from typing import Dict
from langchain_openai import OpenAI

# Initialize LLM
llm = OpenAI()


def llm_agent(state: Dict):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}
