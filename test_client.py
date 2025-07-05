"""
Test Client for ETH Global uAgent
=================================

This client demonstrates how to communicate with the ETH Global uAgent
using the Chat Protocol, simulating ASI:One interactions.
"""

import asyncio
import time
from datetime import datetime, UTC
from uuid import uuid4
from uagents import Agent, Context, Protocol
from uagents_core.contrib.protocols.chat import (
    ChatMessage, ChatAcknowledgement, TextContent, chat_protocol_spec
)

# Test client agent
test_client = Agent(
    name="test_client",
    port=8001,
    mailbox=False,
    seed="test_client_unique_seed_2024"
)

# Chat protocol for communication
chat_proto = Protocol(spec=chat_protocol_spec)

# ETH Global agent address (update this with your deployed agent's address)
ETH_GLOBAL_AGENT_ADDRESS = "agent1q0h70caed8ax769shpemapzkyk65uscw4xwk6dc4t3emvp5jdcvqs9xs32y"

# Test queries to demonstrate different capabilities
TEST_QUERIES = [
    "Analyze the current market trends for Ethereum and provide investment recommendations",
    "What are the best DeFi protocols for yield farming in the current market?",
    "Analyze the Bored Ape Yacht Club NFT collection and provide insights",
    "How can I optimize my smart contract for gas efficiency and security?",
    "Search for the latest developments in blockchain technology and AI integration",
    "Provide a comprehensive analysis of the current crypto market sentiment"
]

@test_client.on_event("startup")
async def startup(ctx: Context):
    """Client startup handler"""
    ctx.logger.info(f"🧪 Test client started: {test_client.address}")
    ctx.logger.info(f"🎯 Target agent: {ETH_GLOBAL_AGENT_ADDRESS}")
    
    # Wait a moment for the target agent to be ready
    await asyncio.sleep(2)
    
    # Start testing
    await run_tests(ctx)

@test_client.on_event("shutdown")
async def shutdown(ctx: Context):
    """Client shutdown handler"""
    ctx.logger.info("🛑 Test client shutting down...")

@chat_proto.on_message(ChatMessage)
async def handle_response(ctx: Context, sender: str, msg: ChatMessage):
    """Handle responses from the ETH Global agent"""
    ctx.logger.info(f"📨 Received response from {sender}")
    
    for item in msg.content:
        if isinstance(item, TextContent):
            ctx.logger.info(f"💬 Response: {item.text[:200]}...")
            if len(item.text) > 200:
                ctx.logger.info(f"   ... (truncated, full response: {len(item.text)} characters)")
    
    # Send acknowledgment
    await ctx.send(sender, ChatAcknowledgement(
        timestamp=datetime.now(UTC),
        acknowledged_msg_id=msg.msg_id
    ))

@chat_proto.on_message(ChatAcknowledgement)
async def handle_acknowledgement(ctx: Context, sender: str, msg: ChatAcknowledgement):
    """Handle acknowledgments"""
    ctx.logger.info(f"✅ Acknowledgment received for message: {msg.acknowledged_msg_id}")

async def run_tests(ctx: Context):
    """Run comprehensive tests of the ETH Global agent"""
    ctx.logger.info("🚀 Starting comprehensive agent tests...")
    
    for i, query in enumerate(TEST_QUERIES, 1):
        ctx.logger.info(f"\n{'='*60}")
        ctx.logger.info(f"🧪 Test {i}/{len(TEST_QUERIES)}")
        ctx.logger.info(f"📝 Query: {query}")
        ctx.logger.info(f"{'='*60}")
        
        # Create and send chat message
        chat_msg = ChatMessage(
            timestamp=datetime.now(UTC),
            msg_id=uuid4(),
            content=[TextContent(type="text", text=query)]
        )
        
        try:
            await ctx.send(ETH_GLOBAL_AGENT_ADDRESS, chat_msg)
            ctx.logger.info(f"📤 Query sent successfully")
            
            # Wait for response
            await asyncio.sleep(5)
            
        except Exception as e:
            ctx.logger.error(f"❌ Failed to send query: {e}")
        
        # Wait between tests
        if i < len(TEST_QUERIES):
            ctx.logger.info("⏳ Waiting 3 seconds before next test...")
            await asyncio.sleep(3)
    
    ctx.logger.info(f"\n{'='*60}")
    ctx.logger.info("🎉 All tests completed!")
    ctx.logger.info(f"{'='*60}")

# Include chat protocol
test_client.include(chat_proto, publish_manifest=True)

if __name__ == "__main__":
    print("""
🧪 ETH Global uAgent Test Client
================================

This client will test the ETH Global uAgent with various queries
to demonstrate its capabilities.

Features tested:
• Market analysis and insights
• DeFi protocol recommendations  
• NFT collection analysis
• Smart contract guidance
• Web search integration
• General AI assistance

🛑 Stop with Ctrl+C
    """)
    
    test_client.run() 