# Chatbot API Server

Flask REST API that serves the customer support chatbot with a React frontend.

## Features

- REST API endpoints for chatbot interaction
- Integrates Python RAG chatbot backend
- Serves React static frontend
- CORS enabled for development
- Health check endpoint

## Setup

### Prerequisites
- Python 3.11+
- Virtual environment (recommended)

### Installation

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Server

```bash
# With virtual environment active
python app.py
```

The API will be available at `http://localhost:5000`

## API Endpoints

### Health Check
- **GET** `/api/health`
- Returns: `{ "status": "ok", "chatbot": "initialized" }`

### Chat
- **POST** `/api/chat`
- Body: `{ "query": "Your question here" }`
- Returns: `{ "response": "Answer from chatbot" }`

### Frontend
- **GET** `/`
- Serves the React application

## Project Structure

```
chatbot-api/
├── app.py              # Flask application
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Development

### With Hot Reload

The Flask app runs in debug mode by default. Changes to `app.py` will automatically reload the server.

### Environment Variables

- `HF_TOKEN` - Hugging Face API token (optional, for higher rate limits)

## Frontend Integration

The React frontend is built and served from `../chatbot-ui/dist/`

To rebuild the frontend:

```bash
cd ../chatbot-ui
npm run build
```

## Performance Notes

- First request will take time as models are loaded into memory
- Chatbot uses 8-bit quantization for memory efficiency
- Responses are generated on-the-fly (not cached)

## Troubleshooting

**ModuleNotFoundError:** Make sure virtual environment is activated
**Port already in use:** Change port in `app.py` or kill process using port 5000
**CUDA errors:** Ensure torch is properly installed with CPU-only or GPU support
