from langchain_openai import ChatOpenAI
from utils.tools import search_google
from langgraph.prebuilt import create_react_agent


def create_research_agent(model="gpt-4o-mini"):
    llm = ChatOpenAI(model=model, temperature=0)

    tools = [search_google]

    return create_react_agent(
        llm,
        tools,
        prompt="You are a research agent focused on finding accurate information. Do not perform calculations."
    )
