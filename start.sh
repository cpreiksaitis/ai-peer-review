#!/bin/sh
# Start script for Railway deployment
# Falls back to port 8000 if PORT is not set

PORT="${PORT:-8000}"
echo "Starting server on port $PORT"
exec uvicorn src.web.production.app:app --host 0.0.0.0 --port "$PORT"

