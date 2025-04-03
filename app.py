# app.py
from flask import Flask
from flask_cors import CORS
from routes.chat_routes import chat_routes
from routes.upload_routes import chat_routes as upload_routes
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_app():
    """Create and configure the Flask application"""
    app = Flask(__name__)
    CORS(app)

    # Register blueprints
    app.register_blueprint(chat_routes, url_prefix='/api')
    app.register_blueprint(upload_routes, url_prefix='/api')

    # Error handlers
    @app.errorhandler(404)
    def not_found(e):
        print("errorhandler ==>", e)
        return {"error": "Resource not found", "status": "error"}, 404

    @app.errorhandler(500)
    def server_error(e):
        print("errorhandler ==>", e)
        return {"error": "Internal server error", "status": "error"}, 500

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
