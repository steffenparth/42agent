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

# Define Cloudflare-Tunnel URL (aus "start_public_agent.sh") hier eintragen:
PUBLIC_ENDPOINT = "https://7b90-83-144-23-155.ngrok-free.app"

# Create the agent with enhanced configuration - discoverable on ASI:One TESTNET
agent = Agent(
    name="ProjectFinderAgent",
    seed="project-finder-123",  # statischer Seed
    endpoint=PUBLIC_ENDPOINT,
    port=8000,
    network="testnet",
    mailbox=True,
    publish_agent_details=True,
)

# Initialize chat handler with LLM function
chat_handler = DefaultChatHandler(get_completion)

# Global client instance - initialized once at startup
global_client = None

@agent.on_event("startup")
async def on_startup(ctx: Context):
    """Enhanced startup handler"""
    global global_client
    
    ctx.logger.info(f"🚀 ProjectFinderAgent started: {agent.address}")
    ctx.logger.info("📡 Agentverse integration ready")
    ctx.logger.info("💬 Enhanced chat protocol loaded")
    ctx.logger.info("🌐 Public endpoint: " + PUBLIC_ENDPOINT)
    
    # Initialize the client once at startup
    try:
        from client import ETHGlobalClient
        global_client = ETHGlobalClient()
        ctx.logger.info("📁 Database initialized successfully")
    except Exception as e:
        ctx.logger.error(f"❌ Failed to initialize database: {e}")
        global_client = None
    
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
    message_count = ctx.storage.get("message_count") or 0
    ctx.logger.info(f"📊 Total messages processed: {message_count}")

@chat_proto.on_message(ChatMessage)
async def handle_message(ctx: Context, sender: str, msg: ChatMessage):
    """Enhanced message handler with session management and error handling"""
    try:
        # Increment message counter
        current_count = ctx.storage.get("message_count") or 0
        ctx.storage.set("message_count", current_count + 1)
        
        # Log incoming message
        message_text = "No content"
        if msg.content and len(msg.content) > 0:
            content = msg.content[0]
            if hasattr(content, 'text'):
                message_text = content.text
            elif hasattr(content, 'type') and content.type == 'text':
                message_text = getattr(content, 'text', 'No text content')
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
        
        # Skip processing for session management messages
        if "Session message:" in message_text or message_text == "No content":
            ctx.logger.info(f"📝 Skipping session message from {sender}")
            return
            
        # Always try semantic search first - let the embedding model decide relevance
        try:
            global global_client
            
            if global_client is None:
                ctx.logger.warning("⚠️  Database not initialized, skipping search")
                raise Exception("Database not available")
            
            # Use the message as the search query
            search_query = message_text
            
            # Perform semantic search with adaptive thresholds
            results = global_client.search_projects(search_query, top_k=5, min_similarity=0.15)
            
            if results:
                response_text = f"🔍 Found {len(results)} relevant projects from ETHGlobal:\n\n"
                for i, project in enumerate(results, 1):
                    response_text += f"{i}. **{project.project_title}**\n"
                    response_text += f"   {project.short_description[:150]}...\n"
                    response_text += f"   🔗 URL: {project.url}\n\n"
                
                response_text += "💡 You can ask me to search for any topic, technology, or project type!"
                await ctx.send(sender, create_text_message(response_text))
                ctx.logger.info(f"✅ Semantic project search response sent to {sender}")
                return
                
        except Exception as e:
            ctx.logger.error(f"Error in project search: {e}")
            # Continue to LLM processing if search fails
        
        # If no relevant projects found or search failed, process with LLM
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
        
        global global_client
        
        if global_client is None:
            await ctx.send(sender, create_error_message("Database not available"))
            return
        
        results = global_client.search_projects(msg.query)
        
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
        
        global global_client
        
        if global_client is None:
            await ctx.send(sender, create_error_message("Database not available"))
            return
        
        agents = await global_client.agentverse_client.search_agents(msg.query, msg.limit)
        
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
    print("🚀 Starting ProjectFinderAgent...")
    print("📡 Agentverse integration: ENABLED")
    print("💬 Enhanced chat protocol: ENABLED")
    print("🔍 Project search: ENABLED")
    print("🤖 Agentverse search: ENABLED")
    print("⏰ Periodic cleanup: ENABLED")
    print("📊 Rate limiting: ENABLED")
    print("🌐 Public endpoint: " + PUBLIC_ENDPOINT)
    print("\nUse Ctrl+C to stop the agent")
    
    try:
        agent.run()
    except KeyboardInterrupt:
        print("\n🛑 Agent stopped by user")
    except Exception as e:
        print(f"💥 Agent crashed: {e}")
 