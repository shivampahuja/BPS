#!/bin/bash
# Quick-start script for local development
set -e

cd "$(dirname "$0")/app"
echo "🚀 Starting CII Web Application on http://localhost:8000"
python -m uvicorn webapp:app --host 0.0.0.0 --port ${PORT:-8000} --reload
