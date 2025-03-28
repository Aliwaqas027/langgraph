from flask import Blueprint, request, jsonify, Response
import logging

from services.graph import GraphService
from utils.tools import frontend_agent_tool, backend_agent_tool, designer_agent_tool, search_google, legal_knowledge_base

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


chat_routes = Blueprint('chat', __name__)


def get_graph_service() -> GraphService:
    try:
        tools = [frontend_agent_tool, backend_agent_tool, designer_agent_tool, search_google, legal_knowledge_base]
        return GraphService(tools)
    except Exception as e:
        logger.error(f"Error initializing graph service: {str(e)}")
        raise


graph_service = get_graph_service()


@chat_routes.route('/chat', methods=['POST'])
async def chat() -> tuple[Response, int] | Response:
    try:
        # Get request data
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({
                'error': 'Missing message in request body'
            }), 400
        logger.info(f"Received chat request: {data['message']}")
        response = await graph_service.process_query(data['message'])

        return jsonify(response)

    except Exception as e:
        # Log error
        logger.error(f"Error processing chat request: {str(e)}")
        return jsonify({
            'error': f'Error processing request: {str(e)}'
        }), 500


