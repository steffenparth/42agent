#!/bin/bash

echo "🚀 Starting ProjectFinderAgent with ngrok tunnel..."

# Change to the agent directory
cd .asi-agent

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Start the agent in the background
echo "🤖 Starting ProjectFinderAgent..."
python ai_agent.py &
AGENT_PID=$!

# Wait a moment for the agent to start
echo "⏳ Waiting for agent to start..."
sleep 8

# Start ngrok tunnel
echo "🌐 Starting ngrok tunnel..."
ngrok http 8000 &
NGROK_PID=$!

# Wait for ngrok to start and get the URL
echo "⏳ Waiting for ngrok to start..."
sleep 5

# Get the ngrok URL
NGROK_URL=$(curl -s http://localhost:4040/api/tunnels | grep -o '"public_url":"[^"]*"' | head -1 | cut -d'"' -f4)

if [ -n "$NGROK_URL" ]; then
    echo ""
    echo "✅ SUCCESS! Your agent is now public!"
    echo "🔗 Public URL: $NGROK_URL"
    echo ""
    echo "📋 Next steps:"
    echo "1. Copy this URL: $NGROK_URL"
    echo "2. Update the PUBLIC_ENDPOINT in ai_agent.py with this URL"
    echo "3. Restart the agent to use the new endpoint"
    echo "4. Register on Agentverse using this URL"
    echo ""
    echo "📱 Agent PID: $AGENT_PID"
    echo "🌐 ngrok PID: $NGROK_PID"
    echo ""
    echo "Press Ctrl+C to stop both processes"
else
    echo "❌ Failed to get ngrok URL. Check if ngrok is running properly."
fi

# Wait for user to stop
wait 