from flask import Blueprint, request, jsonify, Response
import logging

from agents.research_agent import create_research_agent
from agents.backend_agent import se_agent
from agents.frontend_agent import fe_agent
from agents.designer_agent import design_agent
from agents.supervisor import SupervisorService
from agents.response_agent import response_agent
from services.graph import GraphService


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


chat_routes = Blueprint('chat', __name__)


def get_graph_service() -> GraphService:
    try:
        research_agent = create_research_agent()
        backend_agent = se_agent()
        frontend_agent = fe_agent()
        designer_agent = design_agent()
        supervisor = SupervisorService()
        response = response_agent()
        return GraphService(supervisor, research_agent, backend_agent, frontend_agent, designer_agent, response)
    except Exception as e:
        logger.error(f"Error initializing graph service: {str(e)}")
        raise


graph_service = get_graph_service()


@chat_routes.route('/chat', methods=['POST'])
def chat() -> tuple[Response, int] | Response:
    try:
        # Get request data
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({
                'error': 'Missing message in request body'
            }), 400
        logger.info(f"Received chat request: {data['message']}")
        response = graph_service.process_query(data['message'])
        logger.info(f"Generated response for query: {response[:100]}...")

        return jsonify({
            'response': response
        })

    except Exception as e:
        # Log error
        logger.error(f"Error processing chat request: {str(e)}")
        return jsonify({
            'error': f'Error processing request: {str(e)}'
        }), 500


