#!/bin/bash

echo "🚀 Starting ASI Agent with Public Access..."

# Change to the agent directory
cd .asi-agent

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Start the agent in the background
echo "🤖 Starting ASI Agent..."
python ai_agent.py &
AGENT_PID=$!

# Wait a moment for the agent to start
sleep 5

# Start Cloudflare tunnel
echo "🌐 Starting Cloudflare tunnel..."
cloudflared tunnel --url http://localhost:8000 &
TUNNEL_PID=$!

echo "✅ Agent and tunnel started!"
echo "📱 Agent PID: $AGENT_PID"
echo "🌐 Tunnel PID: $TUNNEL_PID"
echo ""
echo "🔗 Your agent will be available at the URL shown above"
echo "📋 Copy that URL and use it in Agentverse registration"
echo ""
echo "Press Ctrl+C to stop both processes"

# Wait for user to stop
wait 