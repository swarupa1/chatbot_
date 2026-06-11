import os
import sys
import json
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from pathlib import Path

# Import the chatbot components from customer support module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ECOM'))

try:
    from customer_support_chatbot_huggingface import (
        documents,
        collection,
        generator,
        get_answer as generate_answer
    )
    CHATBOT_IMPORTED = True
except Exception as e:
    print(f"Warning: Could not import from customer_support_chatbot_huggingface: {e}")
    CHATBOT_IMPORTED = False

app = Flask(__name__, static_folder='../chatbot-ui/dist', static_url_path='')
CORS(app)

# Initialize chatbot components
def initialize_chatbot():
    """Initialize the RAG chatbot system"""
    try:
        if CHATBOT_IMPORTED:
            return {
                'collection': collection,
                'generator': generator,
                'status': 'initialized'
            }
        else:
            return {
                'status': 'error',
                'error': 'Chatbot module not available'
            }
    except Exception as e:
        print(f"Error initializing chatbot: {e}")
        return {
            'status': 'error',
            'error': str(e)
        }

# Initialize on startup
chatbot_state = initialize_chatbot()

def get_answer(query: str) -> str:
    """Generate an answer using RAG"""
    if chatbot_state['status'] != 'initialized':
        return "Chatbot is not initialized"

    try:
        if CHATBOT_IMPORTED:
            return generate_answer(query)
        else:
            return "Chatbot module not available"
    except Exception as e:
        return f"Error generating response: {str(e)}"

# API Routes
@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'chatbot': chatbot_state['status']
    })

@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat endpoint"""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()

        if not query:
            return jsonify({'error': 'Query cannot be empty'}), 400

        response = get_answer(query)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Serve React static files
@app.route('/')
def serve_index():
    """Serve index.html"""
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    """Serve static files"""
    file_path = os.path.join(app.static_folder, path)
    if os.path.isfile(file_path):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
