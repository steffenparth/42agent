"""
Deployment Script for ETH Global uAgent to Agentverse
====================================================

This script deploys the ETH Global uAgent to Agentverse.ai using the uagents-adapter,
making it discoverable through ASI:One at https://asi1.ai
"""

import os
import asyncio
import time
from dotenv import load_dotenv
from uagents_adapter import LangchainRegisterTool, cleanup_uagent

# Load environment variables
load_dotenv()

# Configuration
AGENTVERSE_API_KEY = os.getenv("AGENTVERSE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

def validate_environment():
    """Validate required environment variables"""
    if not AGENTVERSE_API_KEY:
        raise ValueError("AGENTVERSE_API_KEY environment variable is required")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    
    print("✅ Environment variables validated")

def create_agent_function():
    """Create the agent function for deployment"""
    from eth_global_agent import process_user_query_internal
    
    async def agent_function(query):
        """Wrapper function for the agent"""
        if isinstance(query, dict) and 'input' in query:
            query = query['input']
        
        try:
            result = await process_user_query_internal(query)
            return result
        except Exception as e:
            return f"Error processing request: {str(e)}"
    
    return agent_function

async def deploy_agent():
    """Deploy the agent to Agentverse"""
    print("🚀 Starting deployment to Agentverse...")
    
    # Validate environment
    validate_environment()
    
    # Create agent function
    agent_func = create_agent_function()
    
    # Initialize deployment tool
    register_tool = LangchainRegisterTool()
    
    # Deployment configuration
    deployment_config = {
        "agent_obj": agent_func,
        "name": "eth_global_innovator_agent",
        "port": 8000,
        "description": "ETH Global AI Innovation Agent - Multi-modal AI capabilities for blockchain, DeFi, NFTs, and smart contracts. Features web search, market analysis, and intelligent task routing.",
        "api_token": AGENTVERSE_API_KEY,
        "mailbox": True,
        "query_params": {
            "query": {
                "type": "string",
                "description": "User query for AI analysis, market insights, DeFi recommendations, or general assistance",
                "required": True
            }
        },
        "example_query": "Analyze the current market trends for Ethereum and provide DeFi recommendations"
    }
    
    try:
        print("📤 Deploying agent to Agentverse...")
        result = register_tool.invoke(deployment_config)
        
        print(f"✅ Agent deployed successfully!")
        print(f"📋 Deployment result: {result}")
        
        return result
        
    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        raise

async def main():
    """Main deployment function"""
    try:
        result = await deploy_agent()
        
        print("\n" + "="*60)
        print("🎉 ETH Global uAgent Successfully Deployed!")
        print("="*60)
        print(f"📍 Agent Name: eth_global_innovator_agent")
        print(f"🔗 Discoverable on ASI:One: https://asi1.ai")
        print(f"📧 Mailbox Enabled: True")
        print(f"🌐 Agentverse Integration: Active")
        print("="*60)
        
        print("\n📋 Next Steps:")
        print("1. Visit https://asi1.ai to discover your agent")
        print("2. Test the agent with various queries")
        print("3. Monitor performance and usage")
        print("4. Submit your project to ETH Global")
        
        # Keep the deployment alive
        print("\n🔄 Keeping deployment alive... (Press Ctrl+C to stop)")
        while True:
            await asyncio.sleep(60)
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down deployment...")
        try:
            cleanup_uagent("eth_global_innovator_agent")
            print("✅ Agent cleanup completed")
        except Exception as e:
            print(f"⚠️ Cleanup warning: {e}")
    except Exception as e:
        print(f"❌ Deployment error: {e}")
        raise

if __name__ == "__main__":
    print("""
🚀 ETH Global uAgent Deployment Script
=====================================

This script will deploy your ETH Global uAgent to Agentverse.ai
making it discoverable through ASI:One at https://asi1.ai

Requirements:
✅ AGENTVERSE_API_KEY environment variable
✅ OPENAI_API_KEY environment variable  
✅ TAVILY_API_KEY environment variable (optional)

🛑 Stop deployment with Ctrl+C
    """)
    
    asyncio.run(main()) 