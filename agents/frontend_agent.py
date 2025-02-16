from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent


def fe_agent(model="gpt-4o-mini"):
    llm = ChatOpenAI(model=model, temperature=0)

    return create_react_agent(
        llm,
        [],
        prompt="You are a helpful senior frontend engineer who answers questions and gives your opinion when asked."
    )
