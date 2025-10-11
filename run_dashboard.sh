#!/bin/bash

# Crypto Trading Strategy Dashboard Launcher
# This script starts the Streamlit dashboard

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Crypto Trading Strategy Dashboard${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo -e "${YELLOW}Warning: pyproject.toml not found. Make sure you're in the project root.${NC}"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d ".venv" ] && [ ! -d "venv" ]; then
    echo -e "${YELLOW}Virtual environment not found. Creating one...${NC}"
    uv sync
fi

# Activate virtual environment if exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

# Check if Streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo -e "${YELLOW}Streamlit not found. Installing dependencies...${NC}"
    uv sync
fi

# Create exports directory if it doesn't exist
mkdir -p exports

echo -e "${GREEN}Starting dashboard...${NC}"
echo ""
echo -e "Dashboard will be available at: ${BLUE}http://localhost:8501${NC}"
echo -e "Press ${YELLOW}Ctrl+C${NC} to stop the server"
echo ""

# Run Streamlit
uv run streamlit run src/crypto_trader/web/app.py

# If using activated venv instead
# streamlit run src/crypto_trader/web/app.py
