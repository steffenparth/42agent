"""
Example Chat Protocol Usage

This script demonstrates how to use the chat protocol components
to create and send messages in the ASI:One format.
"""

from datetime import datetime
from uuid import uuid4
from chat_proto import ChatMessage, ChatAcknowledgement, TextContent

def create_text_message(text: str) -> ChatMessage:
    """
    Create a text message using the chat protocol.
    
    Args:
        text: The message text content
        
    Returns:
        ChatMessage with the text content
    """
    return ChatMessage(
        timestamp=datetime.now(),
        msg_id=uuid4(),
        content=[TextContent(type="text", text=text)]
    )

def create_acknowledgment(message_id: str) -> ChatAcknowledgement:
    """
    Create an acknowledgment for a received message.
    
    Args:
        message_id: The ID of the message being acknowledged
        
    Returns:
        ChatAcknowledgement for the message
    """
    return ChatAcknowledgement(
        timestamp=datetime.now(),
        acknowledged_msg_id=message_id
    )

def print_message_info(message: ChatMessage, sender: str = "Unknown"):
    """
    Print information about a chat message.
    
    Args:
        message: The chat message to display
        sender: The sender of the message
    """
    print(f"📨 Message from {sender}:")
    print(f"   ID: {message.msg_id}")
    print(f"   Timestamp: {message.timestamp}")
    
    for content in message.content:
        if isinstance(content, TextContent):
            print(f"   Content: {content.text}")
        else:
            print(f"   Content Type: {content.type}")

def print_acknowledgment_info(ack: ChatAcknowledgement, sender: str = "Unknown"):
    """
    Print information about an acknowledgment.
    
    Args:
        ack: The acknowledgment to display
        sender: The sender of the acknowledgment
    """
    print(f"✅ Acknowledgment from {sender}:")
    print(f"   Acknowledged Message ID: {ack.acknowledged_msg_id}")
    print(f"   Timestamp: {ack.timestamp}")

def main():
    """Demonstrate chat protocol usage."""
    print("💬 Chat Protocol Example")
    print("=" * 40)
    
    # Create a sample message
    sample_text = "Hello! This is a test message from the ASI:One agent."
    message = create_text_message(sample_text)
    
    print("📤 Creating a sample message:")
    print_message_info(message, "Example User")
    
    print("\n📥 Creating an acknowledgment:")
    ack = create_acknowledgment(str(message.msg_id))
    print_acknowledgment_info(ack, "ASI:One Agent")
    
    print("\n🔄 Message Flow Example:")
    print("1. Client sends message to agent")
    print("2. Agent sends acknowledgment back")
    print("3. Agent processes message through ASI:One LLM")
    print("4. Agent sends response message to client")
    print("5. Client sends acknowledgment back")
    
    print("\n📋 Message Structure:")
    print(f"ChatMessage:")
    print(f"  - timestamp: {message.timestamp}")
    print(f"  - msg_id: {message.msg_id}")
    print(f"  - content: List of AgentContent objects")
    print(f"    - TextContent: {message.content[0].text}")
    
    print(f"\nChatAcknowledgement:")
    print(f"  - timestamp: {ack.timestamp}")
    print(f"  - acknowledged_msg_id: {ack.acknowledged_msg_id}")
    
    print("\n🎯 This format is compatible with ASI:One's chat protocol!")
    print("   Use these components in your agent implementation.")

if __name__ == "__main__":
    main() 