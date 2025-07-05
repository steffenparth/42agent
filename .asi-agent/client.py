"""
Client for ASI:One Agent Communication

This module provides a client that can communicate with the ASI:One agent
using the chat protocol. It demonstrates how to send messages and receive
responses from the agent.
"""

import asyncio
from datetime import datetime
from uuid import uuid4
from typing import Optional

from uagents import Agent, Context, Protocol
from chat_proto import ChatMessage, ChatAcknowledgement, TextContent, chat_protocol_spec

# Create a client agent
client_agent = Agent(
    name="asi-client",
    seed="client_unique_seed_phrase_2024",
    port=8001,
    mailbox=False,  # Local client doesn't need mailbox
)

# Initialize the chat protocol
chat_proto = Protocol(spec=chat_protocol_spec)

# Store the ASI:One agent address (will be updated when agent starts)
ASI_AGENT_ADDRESS = None

@client_agent.on_event("startup")
async def client_startup(ctx: Context):
    """Handle client startup"""
    ctx.logger.info(f"🚀 ASI:One Client started successfully!")
    ctx.logger.info(f"📍 Client address: {client_agent.address}")
    ctx.logger.info(f"👂 Client is ready to communicate with ASI:One agent")

@client_agent.on_event("shutdown")
async def client_shutdown(ctx: Context):
    """Handle client shutdown"""
    ctx.logger.info("🛑 ASI:One Client shutting down...")

@chat_proto.on_message(ChatMessage)
async def handle_agent_response(ctx: Context, sender: str, msg: ChatMessage):
    """
    Handle responses from the ASI:One agent.
    
    Args:
        ctx: The client context
        sender: The address of the ASI:One agent
        msg: The response message
    """
    try:
        # Extract the text content
        if not msg.content or not isinstance(msg.content[0], TextContent):
            ctx.logger.warning(f"Received message without text content from {sender}")
            return
        
        response_text = msg.content[0].text
        ctx.logger.info(f"🤖 ASI:One Agent Response: {response_text}")
        
        # Send acknowledgment
        ack = ChatAcknowledgement(
            timestamp=datetime.now(),
            acknowledged_msg_id=msg.msg_id
        )
        await ctx.send(sender, ack)
        
    except Exception as e:
        ctx.logger.error(f"❌ Error handling response from {sender}: {str(e)}")

@chat_proto.on_message(ChatAcknowledgement)
async def handle_acknowledgment(ctx: Context, sender: str, msg: ChatAcknowledgement):
    """
    Handle acknowledgments from the ASI:One agent.
    
    Args:
        ctx: The client context
        sender: The address of the ASI:One agent
        msg: The acknowledgment message
    """
    ctx.logger.info(f"✅ Received acknowledgment from {sender} for message: {msg.acknowledged_msg_id}")

# Include the chat protocol in the client
client_agent.include(chat_proto, publish_manifest=True)

async def send_message_to_agent(message: str, agent_address: str) -> None:
    """
    Send a message to the ASI:One agent.
    
    Args:
        message: The message to send
        agent_address: The address of the ASI:One agent
    """
    try:
        # Create the chat message
        chat_msg = ChatMessage(
            timestamp=datetime.now(),
            msg_id=uuid4(),
            content=[TextContent(type="text", text=message)]
        )
        
        # Send the message
        await client_agent.send(agent_address, chat_msg)
        print(f"📤 Sent message to ASI:One agent: {message}")
        
    except Exception as e:
        print(f"❌ Error sending message: {str(e)}")

async def interactive_chat(agent_address: str):
    """
    Start an interactive chat session with the ASI:One agent.
    
    Args:
        agent_address: The address of the ASI:One agent
    """
    print(f"\n💬 Starting interactive chat with ASI:One agent at {agent_address}")
    print("Type your messages (or 'quit' to exit):")
    
    while True:
        try:
            # Get user input
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Send the message
            await send_message_to_agent(user_input, agent_address)
            
            # Wait a bit for the response
            await asyncio.sleep(1)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")

def run_client(agent_address: Optional[str] = None):
    """
    Run the client with optional agent address.
    
    Args:
        agent_address: The address of the ASI:One agent (if None, will prompt)
    """
    if not agent_address:
        agent_address = input("Enter the ASI:One agent address: ").strip()
    
    if not agent_address:
        print("❌ No agent address provided!")
        return
    
    print(f"🔗 Connecting to ASI:One agent at: {agent_address}")
    
    # Start the client agent
    client_agent.run()

async def run_interactive_client(agent_address: Optional[str] = None):
    """
    Run the client in interactive mode.
    
    Args:
        agent_address: The address of the ASI:One agent (if None, will prompt)
    """
    if not agent_address:
        agent_address = input("Enter the ASI:One agent address: ").strip()
    
    if not agent_address:
        print("❌ No agent address provided!")
        return
    
    print(f"🔗 Connecting to ASI:One agent at: {agent_address}")
    
    # Start the client agent in the background
    client_task = asyncio.create_task(client_agent.run())
    
    try:
        # Wait a bit for the client to start
        await asyncio.sleep(2)
        
        # Start interactive chat
        await interactive_chat(agent_address)
        
    finally:
        # Clean up
        client_task.cancel()
        try:
            await client_task
        except asyncio.CancelledError:
            pass

if __name__ == "__main__":
    print("""
🤖 ASI:One Client

This client can communicate with the ASI:One agent using the chat protocol.

📋 Usage:
• Run the ASI:One agent first (python ai_agent.py)
• Copy the agent address from the agent output
• Run this client and provide the agent address
• Start chatting with the ASI:One agent!

🔧 Requirements:
• ASI:One agent must be running
• uagents and uagents-core packages installed

🛑 Stop with Ctrl+C
    """)
    
    # Get the agent address
    agent_address = input("Enter the ASI:One agent address: ").strip()
    
    if not agent_address:
        print("❌ No agent address provided!")
        print("Please run the ASI:One agent first and copy its address.")
        exit(1)
    
    # Run the interactive client
    try:
        asyncio.run(run_interactive_client(agent_address))
    except KeyboardInterrupt:
        print("\n👋 Client stopped by user")
    except Exception as e:
        print(f"❌ Error running client: {str(e)}") 