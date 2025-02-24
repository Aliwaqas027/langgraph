from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent


def response_agent(model="gpt-4o-mini"):
    llm = ChatOpenAI(model=model, temperature=0)

    return create_react_agent(
        llm,
        [],
        prompt="You are a helpful agent that take multiple response and genetrate a combined response and also list "
               "each agent input"
    )
