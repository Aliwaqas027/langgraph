from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ChatResponse:
    """Data structure for chat responses"""
    response: str
    agent_used: str
    metadata: Dict[str, Any] = None
    error: Optional[str] = None


class ChatService:
    def __init__(self, graph):
        """
        Initialize ChatService with a LangGraph instance

        Args:
            graph: Compiled LangGraph instance
        """
        self.graph = graph
        self.conversation_history: Dict[str, List] = {}

    def process_query(self, query: str, session_id: str = "default") -> ChatResponse:
        """
        Process a user query through the multi-agent system.

        Args:
            query: The user's question or request
            session_id: Unique identifier for the conversation session

        Returns:
            ChatResponse object containing the response and metadata
        """
        try:
            logger.info(f"Processing query for session {session_id}: {query}")

            # Get conversation history for this session
            history = self.conversation_history.get(session_id, [])

            # Create system message if this is a new conversation
            if not history:
                system_msg = SystemMessage(content="""You are a helpful assistant with access to multiple tools
                including Google Search and a knowledge base. You will help users by providing accurate and 
                relevant information.""")
                history.append(system_msg)

            # Add user query to history
            user_msg = HumanMessage(content=query)
            history.append(user_msg)

            # Initialize state
            state = {
                "messages": history,
                "agent_type": ""
            }

            # Process through graph
            result = self.graph.invoke(state)

            # Extract response
            ai_response = result["messages"][-1]

            # Update conversation history
            history.append(ai_response)
            self.conversation_history[session_id] = history

            # Create metadata
            metadata = {
                "session_id": session_id,
                "turn_number": len(history) // 2,  # Counting turns based on user messages
                "total_messages": len(history)
            }

            return ChatResponse(
                response=ai_response.content,
                agent_used=result["agent_type"],
                metadata=metadata
            )

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            return ChatResponse(
                response="",
                agent_used="",
                error=str(e)
            )

    def process_query_with_context(
            self,
            query: str,
            context: Dict[str, Any],
            session_id: str = "default"
    ) -> ChatResponse:
        """
        Process a query with additional context information.

        Args:
            query: The user's question or request
            context: Additional context information
            session_id: Unique identifier for the conversation session

        Returns:
            ChatResponse object containing the response and metadata
        """
        try:
            logger.info(f"Processing contextualized query for session {session_id}")

            # Enhance query with context
            enhanced_query = self._enhance_query_with_context(query, context)

            # Process enhanced query
            return self.process_query(enhanced_query, session_id)

        except Exception as e:
            logger.error(f"Error processing query with context: {str(e)}", exc_info=True)
            return ChatResponse(
                response="",
                agent_used="",
                error=str(e)
            )

    def _enhance_query_with_context(self, query: str, context: Dict[str, Any]) -> str:
        """
        Enhance the query with context information.

        Args:
            query: Original query
            context: Context dictionary

        Returns:
            Enhanced query string with context
        """
        try:
            context_elements = []

            # Add user preferences if available
            if "user_preferences" in context:
                prefs = context["user_preferences"]
                if isinstance(prefs, dict):
                    prefs = json.dumps(prefs)
                context_elements.append(f"User preferences: {prefs}")

            # Add interaction history if available
            if "previous_interactions" in context:
                history = context["previous_interactions"]
                if isinstance(history, list):
                    history = " | ".join(history[-3:])  # Last 3 interactions
                context_elements.append(f"Recent interaction history: {history}")

            # Add any custom context
            if "custom_context" in context:
                custom = context["custom_context"]
                if isinstance(custom, dict):
                    custom = json.dumps(custom)
                context_elements.append(f"Additional context: {custom}")

            # Combine everything
            if context_elements:
                context_str = "\n".join(context_elements)
                return f"{query}\n\nContext:\n{context_str}"

            return query

        except Exception as e:
            logger.error(f"Error enhancing query with context: {str(e)}", exc_info=True)
            return query

    def get_conversation_history(self, session_id: str = "default") -> List:
        """
        Retrieve conversation history for a session.

        Args:
            session_id: Unique identifier for the conversation session

        Returns:
            List of conversation messages
        """
        return self.conversation_history.get(session_id, [])

    def clear_history(self, session_id: str = "default"):
        """
        Clear conversation history for a session.

        Args:
            session_id: Unique identifier for the conversation session
        """
        if session_id in self.conversation_history:
            del self.conversation_history[session_id]
            logger.info(f"Cleared conversation history for session {session_id}")
