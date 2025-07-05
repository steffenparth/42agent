"""
ASI:One AI Agent

This module implements an ASI:One compatible agent that uses the chat protocol
for communication and integrates with ASI:One's LLM for generating responses.
"""

import json
from enum import Enum
from typing import Any
from datetime import datetime
from uuid import uuid4

from llm import get_completion, ASIOneError
from chat_proto import ChatMessage, ChatAcknowledgement, TextContent, chat_protocol_spec
from uagents.experimental.quota import QuotaProtocol, RateLimit
from uagents_core.models import ErrorMessage
from uagents import Agent, Context, Protocol, Model

# Initialize the chat protocol
chat_proto = Protocol(spec=chat_protocol_spec)

# Create the agent with ASI:One compatibility
agent = Agent(
    name="asi-agent",
    seed="kfhsqwertyuiopasdfghjklzxcvbnm987654321",
    port=8000,
    mailbox=True,
    publish_agent_details=True,
)

@agent.on_event("startup")
async def on_startup(ctx: Context):
    """Handle agent startup - log the agent address"""
    ctx.logger.info(f"🚀 ASI:One Agent started successfully!")
    ctx.logger.info(f"📍 Agent address: {agent.address}")
    ctx.logger.info(f"🤖 Agent name: {agent.name}")
    ctx.logger.info(f"🌐 Agent inspector: https://agentverse.ai/inspect/?uri=http%3A//127.0.0.1%3A8001&address={agent.address}")

@agent.on_event("shutdown")
async def on_shutdown(ctx: Context):
    """Handle agent shutdown"""
    ctx.logger.info("🛑 ASI:One Agent shutting down...")

@chat_proto.on_message(ChatMessage)
async def handle_message(ctx: Context, sender: str, msg: ChatMessage):
    """
    Handle incoming chat messages from other agents or clients.
    
    This function:
    1. Logs the received message
    2. Stores the sender in session storage
    3. Sends an acknowledgment
    4. Processes the message through ASI:One LLM
    5. Sends the response back
    """
    try:
        # Extract the text content from the message
        if not msg.content or not isinstance(msg.content[0], TextContent):
            ctx.logger.warning(f"Received message without text content from {sender}")
            return
        
        message_text = msg.content[0].text
        ctx.logger.info(f"📨 Got a message from {sender}: {message_text}")
        
        # Store the sender in session storage for future reference
        ctx.storage.set(str(ctx.session), sender)
        
        # Send acknowledgment immediately
        ack = ChatAcknowledgement(
            timestamp=datetime.now(), 
            acknowledged_msg_id=msg.msg_id
        )
        await ctx.send(sender, ack)
        ctx.logger.info(f"✅ Sent acknowledgment to {sender}")
        
        # Process the message and send response
        await send_message(ctx, sender, msg)
        
    except Exception as e:
        ctx.logger.error(f"❌ Error handling message from {sender}: {str(e)}")
        # Send error acknowledgment if possible
        try:
            error_ack = ChatAcknowledgement(
                timestamp=datetime.now(),
                acknowledged_msg_id=msg.msg_id
            )
            await ctx.send(sender, error_ack)
        except:
            pass

async def send_message(ctx: Context, sender: str, msg: ChatMessage):
    """
    Process the message through ASI:One LLM and send the response.
    
    Args:
        ctx: The agent context
        sender: The address of the message sender
        msg: The original chat message
    """
    try:
        # Extract the text content
        message_text = msg.content[0].text
        
        # Get completion from ASI:One LLM
        ctx.logger.info(f"🤖 Processing message through ASI:One LLM...")
        completion = await get_completion(context="", prompt=message_text)
        
        # Extract the response text
        response_text = completion["choices"][0]["message"]["content"]
        ctx.logger.info(f"💬 ASI:One response: {response_text[:100]}...")
        
        # Create and send the response message
        response_msg = ChatMessage(
            timestamp=datetime.now(),
            msg_id=uuid4(),
            content=[TextContent(type="text", text=response_text)]
        )
        
        await ctx.send(sender, response_msg)
        ctx.logger.info(f"📤 Sent response to {sender}")
        
    except ASIOneError as e:
        ctx.logger.error(f"❌ ASI:One API error: {str(e)}")
        # Send error message
        error_msg = ChatMessage(
            timestamp=datetime.now(),
            msg_id=uuid4(),
            content=[TextContent(type="text", text=f"Sorry, I encountered an error: {str(e)}")]
        )
        await ctx.send(sender, error_msg)
        
    except Exception as e:
        ctx.logger.error(f"❌ Unexpected error processing message: {str(e)}")
        # Send generic error message
        error_msg = ChatMessage(
            timestamp=datetime.now(),
            msg_id=uuid4(),
            content=[TextContent(type="text", text="Sorry, I encountered an unexpected error. Please try again.")]
        )
        await ctx.send(sender, error_msg)

@chat_proto.on_message(ChatAcknowledgement)
async def handle_ack(ctx: Context, sender: str, msg: ChatAcknowledgement):
    """
    Handle acknowledgment messages from other agents.
    
    Args:
        ctx: The agent context
        sender: The address of the acknowledgment sender
        msg: The acknowledgment message
    """
    ctx.logger.info(f"✅ Got an acknowledgement from {sender} for message: {msg.acknowledged_msg_id}")

# Include the chat protocol in the agent
agent.include(chat_proto, publish_manifest=True)

if __name__ == "__main__":
    print("""
🤖 Starting ASI:One Agent...

This agent:
• Implements the ASI:One chat protocol
• Integrates with ASI:One's LLM API
• Handles message acknowledgments
• Provides intelligent responses to queries

📋 Features:
• Chat protocol compliance
• ASI:One LLM integration
• Error handling and logging
• Session management
• Mailbox support for Agentverse

🔧 Requirements:
• ASI1_API_KEY environment variable must be set
• uagents and uagents-core packages installed

🛑 Stop with Ctrl+C
    """)
    
    # Check for required environment variable
    import os
    if not os.getenv("ASI1_API_KEY"):
        print("❌ ERROR: ASI1_API_KEY environment variable not set!")
        print("Please set your ASI:One API key:")
        print("export ASI1_API_KEY='your-api-key-here'")
        exit(1)
    
    # Run the agent
    agent.run() 