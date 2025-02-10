# routes/chat_routes.py
from flask import Blueprint, request, jsonify
from services.chat_service import ChatService
from services.graph import build_graph
from typing import Dict, Any
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Blueprint
chat_routes = Blueprint('chat', __name__)

# Initialize the graph and service
graph = build_graph()
chat_service = ChatService(graph)


def validate_request(data: Dict[str, Any]) -> tuple[bool, str]:
    """Validate the incoming request data"""
    if not data:
        return False, "No data provided"

    if 'query' not in data:
        return False, "No query provided"

    if not isinstance(data['query'], str):
        return False, "Query must be a string"

    if len(data['query'].strip()) == 0:
        return False, "Query cannot be empty"

    return True, ""


@chat_routes.route('/chat', methods=['POST'])
def chat():
    """
    Main chat endpoint that processes queries through the multi-agent system

    Request Body:
    {
        "query": "Your question here",
        "session_id": "unique_session_id",  # Optional
        "context": {  # Optional
            "user_preferences": {},
            "previous_interactions": [],
            "custom_context": {}
        }
    }
    """
    try:
        # Get request data
        data = request.get_json()

        # Validate request
        is_valid, error_message = validate_request(data)
        if not is_valid:
            return jsonify({
                "error": error_message,
                "status": "error"
            }), 400

        # Extract data
        query = data['query']
        session_id = data.get('session_id', f"session_{datetime.now().timestamp()}")
        context = data.get('context', {})

        # Log request
        logger.info(f"Received chat request - Session: {session_id}, Query: {query}")

        # Process through service
        if context:
            response = chat_service.process_query_with_context(
                query=query,
                context=context,
                session_id=session_id
            )
        else:
            response = chat_service.process_query(
                query=query,
                session_id=session_id
            )

        # Handle errors from service
        if response.error:
            return jsonify({
                "error": response.error,
                "status": "error"
            }), 500

        # Return successful response
        return jsonify({
            "status": "success",
            "data": {
                "response": response.response,
                "agent_used": response.agent_used,
                "metadata": response.metadata
            }
        }), 200

    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}", exc_info=True)
        return jsonify({
            "error": "Internal server error",
            "status": "error",
            "details": str(e)
        }), 500


@chat_routes.route('/chat/history', methods=['GET'])
def get_history():
    """
    Get conversation history for a session

    Query Parameters:
    - session_id: The session ID to get history for
    """
    try:
        session_id = request.args.get('session_id')

        if not session_id:
            return jsonify({
                "error": "No session_id provided",
                "status": "error"
            }), 400

        history = chat_service.get_conversation_history(session_id)

        return jsonify({
            "status": "success",
            "data": {
                "session_id": session_id,
                "history": [
                    {
                        "content": msg.content,
                        "type": msg.__class__.__name__,
                        "metadata": getattr(msg, 'additional_kwargs', {})
                    }
                    for msg in history
                ]
            }
        }), 200

    except Exception as e:
        logger.error(f"Error retrieving chat history: {str(e)}", exc_info=True)
        return jsonify({
            "error": "Internal server error",
            "status": "error",
            "details": str(e)
        }), 500


@chat_routes.route('/chat/clear', methods=['POST'])
def clear_history():
    """
    Clear conversation history for a session

    Request Body:
    {
        "session_id": "session_to_clear"
    }
    """
    try:
        data = request.get_json()

        if not data or 'session_id' not in data:
            return jsonify({
                "error": "No session_id provided",
                "status": "error"
            }), 400

        session_id = data['session_id']
        chat_service.clear_history(session_id)

        return jsonify({
            "status": "success",
            "message": f"History cleared for session {session_id}"
        }), 200

    except Exception as e:
        logger.error(f"Error clearing chat history: {str(e)}", exc_info=True)
        return jsonify({
            "error": "Internal server error",
            "status": "error",
            "details": str(e)
        }), 500
