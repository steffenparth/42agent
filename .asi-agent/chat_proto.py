"""
Chat Protocol for ASI Agent

This module defines the chat protocol and message handling for the ASI agent.
It provides the foundation for agent-to-agent communication using the Fetch.ai uAgents framework.
"""

import json
from enum import Enum
from typing import Any, List, Optional, Dict
from datetime import datetime
from uuid import uuid4

from uagents_core.contrib.protocols.chat import (
    ChatMessage, 
    ChatAcknowledgement, 
    TextContent, 
    chat_protocol_spec
)
from uagents import Protocol, Context, Model
from pydantic import Field

# Import the chat protocol specification
chat_proto = Protocol(spec=chat_protocol_spec)

class MessageType(str, Enum):
    """Types of messages that can be sent"""
    TEXT = "text"
    IMAGE = "image"
    FILE = "file"
    ERROR = "error"

class ChatSession:
    """Represents a chat session between agents"""
    
    def __init__(self, session_id: str, sender: str):
        self.session_id = session_id
        self.sender = sender
        self.messages: List[ChatMessage] = []
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
    
    def add_message(self, message: ChatMessage):
        """Add a message to the session"""
        self.messages.append(message)
        self.last_activity = datetime.now()
    
    def get_context(self, max_messages: int = 10) -> str:
        """Get conversation context for LLM"""
        recent_messages = self.messages[-max_messages:] if len(self.messages) > max_messages else self.messages
        context = []
        
        for msg in recent_messages:
            if msg.content and len(msg.content) > 0:
                content = msg.content[0]
                if hasattr(content, 'text'):
                    context.append(f"Message: {content.text}")
        
        return "\n".join(context)

class EnhancedChatMessage(Model):
    """Enhanced chat message with additional metadata"""
    message_type: MessageType = Field(default=MessageType.TEXT, description="Type of message")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
    priority: int = Field(default=1, description="Message priority (1-10)")
    requires_response: bool = Field(default=True, description="Whether this message requires a response")

class ChatResponse(Model):
    """Response model for chat interactions"""
    success: bool = Field(description="Whether the operation was successful")
    message: str = Field(description="Response message")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Additional response data")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")

class ChatHandler:
    """Base class for chat message handlers"""
    
    def __init__(self):
        self.sessions: Dict[str, ChatSession] = {}
    
    async def handle_message(self, ctx: Context, sender: str, msg: ChatMessage) -> ChatResponse:
        """Handle incoming chat message"""
        raise NotImplementedError("Subclasses must implement handle_message")
    
    async def send_acknowledgement(self, ctx: Context, sender: str, msg: ChatMessage):
        """Send acknowledgement for received message"""
        await ctx.send(
            sender,
            ChatAcknowledgement(
                timestamp=datetime.now(),
                acknowledged_msg_id=msg.msg_id
            )
        )
    
    def get_or_create_session(self, session_id: str, sender: str) -> ChatSession:
        """Get existing session or create new one"""
        if session_id not in self.sessions:
            self.sessions[session_id] = ChatSession(session_id, sender)
        return self.sessions[session_id]
    
    def cleanup_old_sessions(self, max_age_hours: int = 24):
        """Clean up old chat sessions"""
        cutoff_time = datetime.now().replace(hour=datetime.now().hour - max_age_hours)
        sessions_to_remove = []
        
        for session_id, session in self.sessions.items():
            if session.last_activity < cutoff_time:
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            del self.sessions[session_id]

class DefaultChatHandler(ChatHandler):
    """Default chat message handler that uses LLM for responses"""
    
    def __init__(self, llm_function):
        super().__init__()
        self.llm_function = llm_function
    
    async def handle_message(self, ctx: Context, sender: str, msg: ChatMessage) -> ChatResponse:
        """Handle incoming chat message using LLM"""
        try:
            # Get or create session
            session = self.get_or_create_session(str(ctx.session), sender)
            session.add_message(msg)
            
            # Get conversation context
            context = session.get_context()
            
            # Extract message text
            message_text = ""
            if msg.content and len(msg.content) > 0:
                content = msg.content[0]
                if hasattr(content, 'text'):
                    message_text = content.text
            
            # Get LLM response
            if self.llm_function:
                completion = await self.llm_function(context=context, prompt=message_text)
                
                if completion and "choices" in completion and len(completion["choices"]) > 0:
                    response_text = completion["choices"][0]["message"]["content"]
                    
                    # Send response back
                    await ctx.send(sender, ChatMessage(
                        timestamp=datetime.now(),
                        msg_id=str(uuid4()),
                        content=[TextContent(type="text", text=response_text)]
                    ))
                    
                    return ChatResponse(
                        success=True,
                        message="Response sent successfully",
                        data={"response_text": response_text}
                    )
                else:
                    return ChatResponse(
                        success=False,
                        message="Failed to get LLM response",
                        data={"error": "Invalid LLM response format"}
                    )
            else:
                return ChatResponse(
                    success=False,
                    message="LLM function not available",
                    data={"error": "No LLM function provided"}
                )
                
        except Exception as e:
            return ChatResponse(
                success=False,
                message=f"Error handling message: {str(e)}",
                data={"error": str(e)}
            )

# Protocol decorators for message handling
@chat_proto.on_message(ChatMessage)
async def handle_chat_message(ctx: Context, sender: str, msg: ChatMessage):
    """Handle incoming chat messages"""
    ctx.logger.info(f"Received message from {sender}: {msg.content[0].text if msg.content else 'No content'}")
    
    # Store sender in session
    ctx.storage.set(str(ctx.session), sender)
    
    # Send acknowledgement
    await ctx.send(
        sender,
        ChatAcknowledgement(
            timestamp=datetime.now(),
            acknowledged_msg_id=msg.msg_id
        )
    )
    
    # Note: The actual message handling will be done by the agent that includes this protocol
    # This is just the protocol definition

@chat_proto.on_message(ChatAcknowledgement)
async def handle_chat_acknowledgement(ctx: Context, sender: str, msg: ChatAcknowledgement):
    """Handle chat acknowledgements"""
    ctx.logger.info(f"Received acknowledgement from {sender} for message {msg.acknowledged_msg_id}")

# Utility functions for chat operations
def create_text_message(text: str, message_type: MessageType = MessageType.TEXT) -> ChatMessage:
    """Create a text chat message"""
    return ChatMessage(
        timestamp=datetime.now(),
        msg_id=str(uuid4()),
        content=[TextContent(type="text", text=text)]
    )

def create_error_message(error_text: str) -> ChatMessage:
    """Create an error chat message"""
    return ChatMessage(
        timestamp=datetime.now(),
        msg_id=str(uuid4()),
        content=[TextContent(type="text", text=f"Error: {error_text}")]
    )

def format_chat_context(messages: List[ChatMessage], max_length: int = 1000) -> str:
    """Format chat messages into context string for LLM"""
    context_parts = []
    current_length = 0
    
    for msg in reversed(messages):  # Start from most recent
        if msg.content and len(msg.content) > 0:
            content = msg.content[0]
            if hasattr(content, 'text'):
                text = content.text
                if current_length + len(text) > max_length:
                    break
                context_parts.insert(0, text)
                current_length += len(text)
    
    return "\n".join(context_parts) 