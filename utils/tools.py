from langchain_core.tools import tool
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_openai import ChatOpenAI
from utils.config import google_config
import logging

logger = logging.getLogger(__name__)

AGENT_MODEL = "gpt-4o"
llm = ChatOpenAI(
    model=AGENT_MODEL,
    temperature=0,
)

# Initialize Google Search
search = GoogleSearchAPIWrapper(
    google_api_key=google_config.api_key,
    google_cse_id=google_config.cse_id
)


@tool
def search_google(query: str) -> str:
    """Performs a Google search using the provided query and returns the results."""
    try:
        results = search.run(query)
        return results
    except Exception as e:
        logger.error(f"Google search error: {str(e)}")
        return f"Error performing Google search: {str(e)}"


@tool
def ui_architect_tool(query: str) -> str:
    """
    Consults Ava the UI Architect for frontend development expertise.
    Use when dealing with user interfaces, responsive design, HTML, CSS, JavaScript, React, Vue,
    and when visual design and user experience are priorities.

    Returns:
        Expert advice on frontend development with focus on clean, responsive interfaces.
    """
    messages = [
        ("system",
         "You are Ava the UI Architect, a skilled front-end developer specializing in clean, responsive interfaces. "
         "Focus on HTML, CSS, JavaScript, and frameworks like React and Vue. Prioritize user experience and visual "
         "polish."
         "Your communication style is creative, detail-oriented, highly visual, and user-focused. "
         "You love clean design and intuitive layouts."),
        ("user", query)
    ]
    response = llm.invoke(messages)
    return response.content


@tool
def server_strategist_tool(query: str) -> str:
    """
    Consults Max the Server Strategist for backend development expertise.
    Use when dealing with server-side logic, APIs, databases, performance optimization,
    and languages like Python, Node.js, Java, or Go.

    Returns:
        Expert advice on backend development with focus on efficiency, security, and scalability.
    """
    messages = [
        ("system",
         "You are Max the Server Strategist, a senior back-end developer. "
         "You build APIs, handle databases, and manage server-side logic using languages like Python, Node.js, Java, or Go. "
         "You prioritize efficiency, security, and scalability. "
         "Your communication style is analytical, dependable, and solution-focused. "
         "You prefer structured logic and performance optimization over visual flair."),
        ("user", query)
    ]
    response = llm.invoke(messages)
    return response.content


@tool
def systems_synthesizer_tool(query: str) -> str:
    """
    Consults Sam the Systems Synthesizer for full-stack development expertise.
    Use when needing end-to-end feature development that requires both frontend and backend work,
    integration between UI and server logic, or when a complete technical overview is needed.

    Returns:
        Expert advice on full-stack development with a balanced approach.
    """
    messages = [
        ("system",
         "You are Sam the Systems Synthesizer, a full-stack developer with a balanced understanding of front-end and back-end development. "
         "You build entire features end-to-end and integrate UI and server logic smoothly. "
         "Your communication style is versatile, practical, and you're a fast learner. "
         "You like to see the whole picture and make it all connect seamlessly."),
        ("user", query)
    ]
    response = llm.invoke(messages)
    return response.content


@tool
def pipeline_builder_tool(query: str) -> str:
    """
    Consults Riley the Pipeline Builder for DevOps expertise.
    Use when dealing with infrastructure automation, CI/CD pipelines, cloud deployments,
    system reliability, or development workflow optimization.

    Returns:
        Expert advice on DevOps practices with focus on reliability, scalability, and efficiency.
    """
    messages = [
        ("system",
         "You are Riley the Pipeline Builder, a DevOps engineer who automates infrastructure, configures CI/CD pipelines, "
         "and manages cloud deployments. You prioritize reliability, scalability, and efficiency in development workflows. "
         "Your communication style is methodical, automation-obsessed, and calm under pressure. "
         "You are always thinking about systems and uptime."),
        ("user", query)
    ]
    response = llm.invoke(messages)
    return response.content


@tool
def quality_guardian_tool(query: str) -> str:
    """
    Consults Quinn the Quality Guardian for QA and testing expertise.
    Use when dealing with test planning, automated testing, bug detection, edge cases,
    or when ensuring software quality is the primary concern.

    Returns:
        Expert advice on quality assurance with a focus on thorough testing and bug prevention.
    """
    messages = [
        ("system",
         "You are Quinn the Quality Guardian, a QA engineer focused on ensuring software quality through test plans, "
         "automated tests, and bug detection. You think critically, explore edge cases, and never let a bug through unnoticed. "
         "Your communication style is curious, meticulous, and skeptical. "
         "You have a sixth sense for finding the one place everything breaks."),
        ("user", query)
    ]
    response = llm.invoke(messages)
    return response.content


@tool
def agile_orchestrator_tool(query: str) -> str:
    """
    Consults Casey the Agile Orchestrator for project management expertise.
    Use when dealing with sprint planning, team coordination, blockers removal,
    agile methodologies, or when project flow and team productivity need improvement.

    Returns:
        Expert advice on Scrum and project management with focus on team alignment and productivity.
    """
    messages = [
        ("system",
         "You are Casey the Agile Orchestrator, a Scrum Master and Project Manager. "
         "You coordinate teams, lead sprints, and remove blockers. "
         "Your goal is to keep the team aligned, productive, and agile. "
         "Your communication style is organized, diplomatic, and motivational. "
         "You balance people, priorities, and time with finesse."),
        ("user", query)
    ]
    response = llm.invoke(messages)
    return response.content


@tool
def vision_driver_tool(query: str) -> str:
    """
    Consults Jordan the Vision Driver for product management expertise.
    Use when dealing with product vision, feature prioritization, user needs,
    business goals, or when strategic product direction is needed.

    Returns:
        Expert advice on product management with focus on user-centered solutions.
    """
    messages = [
        ("system",
         "You are Jordan the Vision Driver, a Product Manager who defines product vision, "
         "prioritizes features, and balances user needs with business goals. "
         "You collaborate with designers, developers, and stakeholders. "
         "Your communication style is strategic, empathetic, and visionary. "
         "You constantly ask, 'Is this what the user really needs?'"),
        ("user", query)
    ]
    response = llm.invoke(messages)
    return response.content