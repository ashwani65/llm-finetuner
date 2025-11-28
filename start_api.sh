#!/bin/bash

# Startup script for LLM Fine-tuner Production API
# This script initializes the database and starts the API server

set -e  # Exit on error

echo "======================================================"
echo "LLM Fine-tuner Production API - Startup"
echo "======================================================"

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Virtual environment not activated!"
    echo "Please run: source venv/bin/activate"
    exit 1
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p data/uploads
mkdir -p models
mkdir -p logs

# Initialize database (creates tables if they don't exist)
echo "Initializing database..."
python -c "from src.utils.database import get_db; db = get_db(); print('✅ Database initialized')"

# Check if sample dataset exists
if [ ! -f "data/sample_sql_dataset.json" ]; then
    echo "⚠️  Sample dataset not found at data/sample_sql_dataset.json"
    echo "You can create one or upload your own dataset through the API"
fi

# Start the API server
echo ""
echo "======================================================"
echo "Starting Production API Server"
echo "======================================================"
echo ""
python -m src.serving.production_api
