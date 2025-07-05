"""
Chat Protocol for ASI:One Agent Communication

This module provides the chat protocol implementation for ASI:One compatible agents.
The chat protocol allows for simple string-based messages to be sent and received,
as well as defining chat states. It's the expected communication format for ASI:One.
"""

from uagents_core.contrib.protocols.chat import (
    AgentContent, 
    ChatMessage, 
    ChatAcknowledgement, 
    TextContent,
    chat_protocol_spec
)

# Export the chat protocol components
__all__ = [
    'AgentContent',
    'ChatMessage', 
    'ChatAcknowledgement',
    'TextContent',
    'chat_protocol_spec'
]