#!/usr/bin/env bash

# Activate virtual environment
source venv/Scripts/activate  # Use venv/bin/activate on Linux/Mac

# Set PYTHONPATH
export PYTHONPATH=./backend

# Start agents in background
python backend/agents/finance_agent.py &
python backend/agents/logistics_agent.py &
python backend/agents/governance_agent.py &

# Wait for all background processes
wait

echo "Cognify agents started."
