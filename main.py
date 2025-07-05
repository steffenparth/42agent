"""
ETH Global Fetch.ai uAgent - Main Entry Point
============================================

This is the main entry point for the ETH Global uAgent project.
It provides options to run the agent locally or deploy to Agentverse.
"""

import sys
import os
from pathlib import Path

def print_banner():
    """Print project banner"""
    print("""
🚀 ETH Global Fetch.ai uAgent - AI Innovation Agent
==================================================

A comprehensive AI agent built on Fetch.ai's uAgents framework that meets
all ETH Global requirements and demonstrates innovative AI capabilities.

✅ ETH Global Requirements Met:
   • Created uAgent using Fetch.ai framework
   • Hosted on Agentverse.ai with mailbox feature
   • Discoverable through ASI:One at https://asi1.ai
   • Implements Agent Chat Protocol for ASI:One compatibility
   • Public GitHub repository with comprehensive documentation

🌟 Key Features:
   • Multi-modal AI capabilities (text, analysis, recommendations)
   • Web search integration via Tavily
   • Intelligent task routing and execution
   • Real-time market analysis and insights
   • DeFi protocol recommendations
   • NFT and blockchain analytics
   • Smart contract interaction guidance
    """)

def print_menu():
    """Print main menu"""
    print("""
📋 Available Options:
====================

1. 🚀 Run agent locally
2. 🌐 Deploy to Agentverse
3. 🧪 Run test client
4. ⚙️  Setup environment
5. 📚 View documentation
6. 🛑 Exit

Enter your choice (1-6): """)

def run_agent_locally():
    """Run the agent locally"""
    print("\n🚀 Starting ETH Global uAgent locally...")
    try:
        from eth_global_agent import agent
        agent.run()
    except ImportError as e:
        print(f"❌ Error: {e}")
        print("   Please run setup first: python setup.py")
    except KeyboardInterrupt:
        print("\n🛑 Agent stopped by user")

def deploy_to_agentverse():
    """Deploy the agent to Agentverse"""
    print("\n🌐 Deploying to Agentverse...")
    try:
        import asyncio
        from deploy_to_agentverse import main as deploy_main
        asyncio.run(deploy_main())
    except ImportError as e:
        print(f"❌ Error: {e}")
        print("   Please run setup first: python setup.py")
    except KeyboardInterrupt:
        print("\n🛑 Deployment stopped by user")

def run_test_client():
    """Run the test client"""
    print("\n🧪 Starting test client...")
    try:
        from test_client import test_client
        test_client.run()
    except ImportError as e:
        print(f"❌ Error: {e}")
        print("   Please run setup first: python setup.py")
    except KeyboardInterrupt:
        print("\n🛑 Test client stopped by user")

def setup_environment():
    """Run setup script"""
    print("\n⚙️  Running setup...")
    try:
        from setup import main as setup_main
        setup_main()
    except ImportError as e:
        print(f"❌ Error: {e}")
        print("   Please ensure setup.py exists")

def view_documentation():
    """View documentation"""
    print("\n📚 Documentation:")
    print("=================")
    print("• README.md - Comprehensive project documentation")
    print("• fetchai.mdc - Fetch.ai development rules and patterns")
    print("• https://innovationlab.fetch.ai/ - Official Fetch.ai docs")
    print("• https://asi1.ai - ASI:One discovery platform")
    
    # Check if README exists
    if Path("README.md").exists():
        print("\n📖 Opening README.md...")
        try:
            import subprocess
            if sys.platform.startswith('darwin'):  # macOS
                subprocess.run(['open', 'README.md'])
            elif sys.platform.startswith('win'):   # Windows
                subprocess.run(['start', 'README.md'], shell=True)
            else:                                   # Linux
                subprocess.run(['xdg-open', 'README.md'])
        except Exception as e:
            print(f"   Could not open README.md automatically: {e}")
            print("   Please open README.md manually")

def main():
    """Main function"""
    print_banner()
    
    while True:
        print_menu()
        choice = input().strip()
        
        if choice == '1':
            run_agent_locally()
        elif choice == '2':
            deploy_to_agentverse()
        elif choice == '3':
            run_test_client()
        elif choice == '4':
            setup_environment()
        elif choice == '5':
            view_documentation()
        elif choice == '6':
            print("\n👋 Thank you for using ETH Global uAgent!")
            print("   Good luck with your ETH Global submission! 🚀")
            break
        else:
            print("\n❌ Invalid choice. Please enter a number between 1-6.")
        
        print("\n" + "="*60)

if __name__ == "__main__":
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
