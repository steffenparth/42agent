import json
from enum import Enum
from typing import Any, Dict
from datetime import datetime
from llm import get_completion
from uagents_core.contrib.protocols.chat import ChatMessage, ChatAcknowledgement, TextContent, chat_protocol_spec
from uagents.experimental.quota import QuotaProtocol, RateLimit
from uagents_core.models import ErrorMessage
from uagents import Agent, Context, Protocol, Model
from uuid import uuid4
from pydantic import Field

# Import our enhanced chat protocol
from chat_proto import (
    chat_proto, 
    DefaultChatHandler, 
    ChatSession, 
    create_text_message, 
    create_error_message,
    format_chat_context,
    MessageType
)

# Create the agent with enhanced configuration
agent = Agent(
    name="asi-agent-enhanced",
    seed="b7f8c2d9e5a4f1c3d6e7b8a9c0f2e1d4",
    port=8000,
    mailbox=True,
    publish_agent_details=True,
)

# Initialize chat handler with LLM function
chat_handler = DefaultChatHandler(get_completion)

@agent.on_event("startup")
async def on_startup(ctx: Context):
    """Enhanced startup handler"""
    ctx.logger.info(f"🚀 ASI Agent started: {agent.address}")
    ctx.logger.info("📡 Agentverse integration ready")
    ctx.logger.info("💬 Enhanced chat protocol loaded")
    
    # Initialize any required resources
    ctx.storage.set("startup_time", datetime.now().isoformat())
    ctx.storage.set("message_count", 0)

@agent.on_event("shutdown")
async def on_shutdown(ctx: Context):
    """Enhanced shutdown handler"""
    ctx.logger.info("🛑 ASI Agent shutting down...")
    
    # Cleanup old sessions
    chat_handler.cleanup_old_sessions()
    
    # Log final statistics
    message_count = ctx.storage.get("message_count", 0)
    ctx.logger.info(f"📊 Total messages processed: {message_count}")

@chat_proto.on_message(ChatMessage)
async def handle_message(ctx: Context, sender: str, msg: ChatMessage):
    """Enhanced message handler with session management and error handling"""
    try:
        # Increment message counter
        current_count = ctx.storage.get("message_count", 0)
        ctx.storage.set("message_count", current_count + 1)
        
        # Log incoming message
        message_text = msg.content[0].text if msg.content else "No content"
        ctx.logger.info(f"📨 Message #{current_count + 1} from {sender}: {message_text[:50]}...")
        
        # Store sender in session
        ctx.storage.set(str(ctx.session), sender)
        
        # Send immediate acknowledgement
        await ctx.send(
            sender,
            ChatAcknowledgement(
                timestamp=datetime.now(),
                acknowledged_msg_id=msg.msg_id
            )
        )
        
        # Process message with enhanced handler
        response = await chat_handler.handle_message(ctx, sender, msg)
        
        if response.success:
            ctx.logger.info(f"✅ Response sent successfully to {sender}")
        else:
            ctx.logger.error(f"❌ Failed to process message: {response.message}")
            # Send error message to user
            await ctx.send(sender, create_error_message(response.message))
            
    except Exception as e:
        ctx.logger.error(f"💥 Error in message handler: {str(e)}")
        # Send error message to user
        await ctx.send(sender, create_error_message(f"Internal error: {str(e)}"))

@chat_proto.on_message(ChatAcknowledgement)
async def handle_ack(ctx: Context, sender: str, msg: ChatAcknowledgement):
    """Enhanced acknowledgement handler"""
    ctx.logger.info(f"✅ Acknowledgement from {sender} for message {msg.acknowledged_msg_id}")

# Add custom message handlers for specific functionality
class ProjectSearchRequest(Model):
    """Model for project search requests"""
    query: str
    category: str = None
    limit: int = 10

class AgentverseSearchRequest(Model):
    """Model for Agentverse search requests"""
    query: str
    limit: int = 5

@agent.on_message(ProjectSearchRequest)
async def handle_project_search(ctx: Context, sender: str, msg: ProjectSearchRequest):
    """Handle project search requests"""
    try:
        ctx.logger.info(f"🔍 Project search request from {sender}: {msg.query}")
        
        # Import client here to avoid circular imports
        from client import ETHGlobalClient
        
        client = ETHGlobalClient()
        results = client.search_projects(msg.query)
        
        if msg.category:
            from client import ProjectCategory
            try:
                category = ProjectCategory(msg.category)
                results = client.filter_by_category(category)
                # Further filter by query
                results = [p for p in results if msg.query.lower() in 
                          f"{p.project_title} {p.short_description}".lower()]
            except ValueError:
                pass
        
        # Limit results
        results = results[:msg.limit]
        
        # Format response
        response_text = f"Found {len(results)} projects for '{msg.query}':\n\n"
        for i, project in enumerate(results, 1):
            response_text += f"{i}. **{project.project_title}**\n"
            response_text += f"   {project.short_description[:100]}...\n"
            response_text += f"   URL: {project.url}\n\n"
        
        await ctx.send(sender, create_text_message(response_text))
        
    except Exception as e:
        ctx.logger.error(f"Error in project search: {e}")
        await ctx.send(sender, create_error_message(f"Search failed: {str(e)}"))

@agent.on_message(AgentverseSearchRequest)
async def handle_agentverse_search(ctx: Context, sender: str, msg: AgentverseSearchRequest):
    """Handle Agentverse search requests"""
    try:
        ctx.logger.info(f"🤖 Agentverse search request from {sender}: {msg.query}")
        
        # Import client here to avoid circular imports
        from client import ETHGlobalClient
        
        client = ETHGlobalClient()
        agents = await client.agentverse_client.search_agents(msg.query, msg.limit)
        
        # Format response
        response_text = f"Found {len(agents)} agents for '{msg.query}':\n\n"
        for i, agent_data in enumerate(agents, 1):
            response_text += f"{i}. **{agent_data.get('name', 'Unknown')}**\n"
            response_text += f"   {agent_data.get('description', 'No description')[:100]}...\n"
            response_text += f"   ID: {agent_data.get('id', 'Unknown')}\n\n"
        
        await ctx.send(sender, create_text_message(response_text))
        
    except Exception as e:
        ctx.logger.error(f"Error in Agentverse search: {e}")
        await ctx.send(sender, create_error_message(f"Agentverse search failed: {str(e)}"))

# Add periodic cleanup task
@agent.on_interval(period=3600)  # Run every hour
async def periodic_cleanup(ctx: Context):
    """Periodic cleanup of old sessions and resources"""
    try:
        chat_handler.cleanup_old_sessions()
        ctx.logger.info("🧹 Periodic cleanup completed")
    except Exception as e:
        ctx.logger.error(f"Error in periodic cleanup: {e}")

# Include the enhanced chat protocol
agent.include(chat_proto, publish_manifest=True)

# Add quota management for rate limiting
quota_proto = QuotaProtocol(
    RateLimit(
        max_requests=100,  # Max 100 requests
        window_size_minutes=60  # Per hour (60 minutes)
    )
)
agent.include(quota_proto)

if __name__ == "__main__":
    print("🚀 Starting Enhanced ASI Agent...")
    print("📡 Agentverse integration: ENABLED")
    print("💬 Enhanced chat protocol: ENABLED")
    print("🔍 Project search: ENABLED")
    print("🤖 Agentverse search: ENABLED")
    print("⏰ Periodic cleanup: ENABLED")
    print("📊 Rate limiting: ENABLED")
    print("\nUse Ctrl+C to stop the agent")
    
    try:
        agent.run()
    except KeyboardInterrupt:
        print("\n🛑 Agent stopped by user")
    except Exception as e:
        print(f"💥 Agent crashed: {e}")
 