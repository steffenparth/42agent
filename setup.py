#!/usr/bin/env python3
"""
Setup Script for ETH Global uAgent
==================================

This script helps set up the environment for the ETH Global uAgent,
validating dependencies and configuration.
"""

import os
import sys
import subprocess
import getpass
from pathlib import Path

def print_banner():
    """Print setup banner"""
    print("""
🚀 ETH Global Fetch.ai uAgent Setup
===================================

This script will help you set up your environment for the ETH Global uAgent.
    """)

def check_python_version():
    """Check Python version compatibility"""
    print("🐍 Checking Python version...")
    
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        print(f"   Current version: {sys.version}")
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def setup_environment_file():
    """Set up environment file"""
    print("\n🔧 Setting up environment configuration...")
    
    env_file = Path(".env")
    env_example = Path("env.example")
    
    if env_file.exists():
        print("⚠️  .env file already exists")
        overwrite = input("   Do you want to overwrite it? (y/N): ").lower().strip()
        if overwrite != 'y':
            print("   Skipping environment file setup")
            return True
    
    # Copy example file
    if env_example.exists():
        with open(env_example, 'r') as f:
            content = f.read()
        
        with open(env_file, 'w') as f:
            f.write(content)
        
        print("✅ Created .env file from template")
        print("   Please edit .env with your API keys")
        return True
    else:
        print("❌ env.example file not found")
        return False

def get_api_keys():
    """Get API keys from user"""
    print("\n🔑 Setting up API keys...")
    
    api_keys = {}
    
    # Agentverse API Key
    agentverse_key = getpass.getpass("Enter your Fetch.ai Agentverse API Key: ").strip()
    if agentverse_key:
        api_keys['AGENTVERSE_API_KEY'] = agentverse_key
        print("✅ Agentverse API Key configured")
    else:
        print("⚠️  Agentverse API Key not provided")
    
    # OpenAI API Key
    openai_key = getpass.getpass("Enter your OpenAI API Key: ").strip()
    if openai_key:
        api_keys['OPENAI_API_KEY'] = openai_key
        print("✅ OpenAI API Key configured")
    else:
        print("⚠️  OpenAI API Key not provided")
    
    # Tavily API Key (optional)
    tavily_key = getpass.getpass("Enter your Tavily API Key (optional): ").strip()
    if tavily_key:
        api_keys['TAVILY_API_KEY'] = tavily_key
        print("✅ Tavily API Key configured")
    else:
        print("ℹ️  Tavily API Key not provided (web search will be limited)")
    
    return api_keys

def update_env_file(api_keys):
    """Update .env file with API keys"""
    if not api_keys:
        return True
    
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ .env file not found")
        return False
    
    try:
        with open(env_file, 'r') as f:
            content = f.read()
        
        # Update with provided keys
        for key, value in api_keys.items():
            content = content.replace(f"{key}=your_{key.lower()}_here", f"{key}={value}")
        
        with open(env_file, 'w') as f:
            f.write(content)
        
        print("✅ Updated .env file with API keys")
        return True
    except Exception as e:
        print(f"❌ Failed to update .env file: {e}")
        return False

def validate_setup():
    """Validate the setup"""
    print("\n🔍 Validating setup...")
    
    # Check if .env exists
    if not Path(".env").exists():
        print("❌ .env file not found")
        return False
    
    # Check required environment variables
    required_vars = ['AGENTVERSE_API_KEY', 'OPENAI_API_KEY']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("   Please set them in your .env file or environment")
        return False
    
    print("✅ Environment variables validated")
    
    # Test imports
    try:
        import uagents
        import langchain_openai
        import uagents_adapter
        print("✅ All required packages imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    return True

def run_quick_test():
    """Run a quick test of the agent"""
    print("\n🧪 Running quick test...")
    
    try:
        # Test agent creation
        from uagents import Agent
        test_agent = Agent(
            name="test_setup_agent",
            seed="test_setup_seed",
            port=8002
        )
        print("✅ Agent creation test passed")
        
        # Test LangGraph setup
        from langchain_openai import ChatOpenAI
        from langgraph.prebuilt import chat_agent_executor
        
        model = ChatOpenAI(temperature=0)
        app = chat_agent_executor.create_tool_calling_executor(model, [])
        print("✅ LangGraph setup test passed")
        
        return True
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False

def main():
    """Main setup function"""
    print_banner()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Setup failed at dependency installation")
        sys.exit(1)
    
    # Setup environment file
    if not setup_environment_file():
        print("❌ Setup failed at environment file creation")
        sys.exit(1)
    
    # Get API keys
    api_keys = get_api_keys()
    
    # Update .env file
    if api_keys and not update_env_file(api_keys):
        print("❌ Setup failed at API key configuration")
        sys.exit(1)
    
    # Validate setup
    if not validate_setup():
        print("❌ Setup validation failed")
        sys.exit(1)
    
    # Run quick test
    if not run_quick_test():
        print("❌ Quick test failed")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("🎉 Setup completed successfully!")
    print("="*60)
    print("\n📋 Next steps:")
    print("1. Run the agent locally: python eth_global_agent.py")
    print("2. Deploy to Agentverse: python deploy_to_agentverse.py")
    print("3. Test with client: python test_client.py")
    print("4. Visit https://asi1.ai to discover your agent")
    print("\n📚 Documentation: README.md")
    print("🔗 Support: https://innovationlab.fetch.ai/")

if __name__ == "__main__":
    main() 