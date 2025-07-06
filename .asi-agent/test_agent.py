#!/usr/bin/env python3
"""
Test script for the ASI Agent

This script tests all components of the ASI agent system:
- LLM integration
- Chat protocol
- Agent functionality
- Client integration
"""

import asyncio
import sys
import os

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

async def test_llm():
    """Test LLM integration"""
    print("🧪 Testing LLM Integration...")
    
    try:
        from llm import get_completion, test_llm
        success = await test_llm()
        if success:
            print("✅ LLM test passed!")
        else:
            print("❌ LLM test failed!")
        return success
    except Exception as e:
        print(f"❌ LLM test error: {e}")
        return False

async def test_chat_protocol():
    """Test chat protocol functionality"""
    print("\n💬 Testing Chat Protocol...")
    
    try:
        from chat_proto import (
            create_text_message, 
            create_error_message, 
            format_chat_context,
            MessageType
        )
        
        # Test message creation
        text_msg = create_text_message("Hello, world!")
        error_msg = create_error_message("Test error")
        
        print(f"✅ Text message created: {text_msg.content[0].text}")
        print(f"✅ Error message created: {error_msg.content[0].text}")
        
        # Test context formatting
        context = format_chat_context([text_msg, error_msg])
        print(f"✅ Context formatting works: {len(context)} characters")
        
        return True
    except Exception as e:
        print(f"❌ Chat protocol test error: {e}")
        return False

async def test_client():
    """Test client functionality"""
    print("\n📊 Testing Client Integration...")
    
    try:
        from client import ETHGlobalClient
        
        client = ETHGlobalClient()
        
        # Test basic functionality
        stats = client.get_statistics()
        print(f"✅ Client loaded {stats.get('total_projects', 0)} projects")
        
        # Test search functionality
        if stats.get('total_projects', 0) > 0:
            results = client.search_projects("defi", limit=3)
            print(f"✅ Search found {len(results)} DeFi projects")
        
        return True
    except Exception as e:
        print(f"❌ Client test error: {e}")
        return False

async def test_agent_creation():
    """Test agent creation without running it"""
    print("\n🤖 Testing Agent Creation...")
    
    try:
        from ai_agent import agent
        
        print(f"✅ Agent created successfully: {agent.name}")
        print(f"✅ Agent address: {agent.address}")
        print(f"✅ Agent port: {agent.port}")
        
        return True
    except Exception as e:
        print(f"❌ Agent creation test error: {e}")
        return False

async def test_agentverse_integration():
    """Test Agentverse integration"""
    print("\n🌐 Testing Agentverse Integration...")
    
    try:
        from client import AgentverseClient
        
        client = AgentverseClient()
        
        # Test agent search
        agents = await client.search_agents("defi", 3)
        print(f"✅ Found {len(agents)} DeFi agents in Agentverse")
        
        return True
    except Exception as e:
        print(f"❌ Agentverse test error: {e}")
        return False

async def run_all_tests():
    """Run all tests"""
    print("🚀 Starting ASI Agent Test Suite")
    print("=" * 50)
    
    tests = [
        ("LLM Integration", test_llm),
        ("Chat Protocol", test_chat_protocol),
        ("Client Integration", test_client),
        ("Agent Creation", test_agent_creation),
        ("Agentverse Integration", test_agentverse_integration),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = await test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1
    
    print("=" * 50)
    print(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your ASI agent is ready to run.")
        print("\nTo start the agent, run:")
        print("cd .asi-agent && python ai_agent.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total

def main():
    """Main function"""
    try:
        success = asyncio.run(run_all_tests())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"💥 Test suite crashed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 