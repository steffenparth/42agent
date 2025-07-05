"""
Test Setup Script for ASI:One Agent

This script verifies that all dependencies are properly installed
and the environment is configured correctly.
"""

import sys
import os
from typing import List, Tuple

def test_imports() -> List[Tuple[str, bool, str]]:
    """Test if all required packages can be imported."""
    results = []
    
    packages = [
        ("uagents", "uAgents framework"),
        ("uagents_core", "uAgents core components"),
        ("requests", "HTTP requests library"),
        ("pydantic", "Data validation"),
        ("asyncio", "Async support"),
    ]
    
    for package, description in packages:
        try:
            __import__(package)
            results.append((package, True, f"✅ {description} imported successfully"))
        except ImportError as e:
            results.append((package, False, f"❌ {description} import failed: {e}"))
    
    return results

def test_environment() -> List[Tuple[str, bool, str]]:
    """Test environment configuration."""
    results = []
    
    # Test ASI:One API key
    api_key = os.getenv("ASI1_API_KEY")
    if api_key:
        results.append(("ASI1_API_KEY", True, "✅ ASI:One API key is set"))
    else:
        results.append(("ASI1_API_KEY", False, "❌ ASI:One API key not set (export ASI1_API_KEY='your-key')"))
    
    return results

def test_chat_protocol() -> List[Tuple[str, bool, str]]:
    """Test chat protocol imports."""
    results = []
    
    try:
        from chat_proto import ChatMessage, ChatAcknowledgement, TextContent, chat_protocol_spec
        results.append(("chat_proto", True, "✅ Chat protocol imports successful"))
    except ImportError as e:
        results.append(("chat_proto", False, f"❌ Chat protocol imports failed: {e}"))
    
    return results

def test_llm_module() -> List[Tuple[str, bool, str]]:
    """Test LLM module imports."""
    results = []
    
    try:
        from llm import get_completion, ASIOneError
        results.append(("llm", True, "✅ LLM module imports successful"))
    except ImportError as e:
        results.append(("llm", False, f"❌ LLM module imports failed: {e}"))
    
    return results

def test_agent_module() -> List[Tuple[str, bool, str]]:
    """Test agent module imports."""
    results = []
    
    try:
        from ai_agent import agent, chat_proto
        results.append(("ai_agent", True, "✅ Agent module imports successful"))
    except ImportError as e:
        results.append(("ai_agent", False, f"❌ Agent module imports failed: {e}"))
    
    return results

def main():
    """Run all tests and display results."""
    print("🧪 Testing ASI:One Agent Setup")
    print("=" * 50)
    
    all_results = []
    
    # Run all tests
    all_results.extend(test_imports())
    all_results.extend(test_environment())
    all_results.extend(test_chat_protocol())
    all_results.extend(test_llm_module())
    all_results.extend(test_agent_module())
    
    # Display results
    passed = 0
    failed = 0
    
    for test_name, success, message in all_results:
        print(message)
        if success:
            passed += 1
        else:
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Your ASI:One agent is ready to run.")
        print("\n🚀 Next steps:")
        print("1. Start the agent: python ai_agent.py")
        print("2. In another terminal, start the client: python client.py")
        print("3. Enter the agent address when prompted")
        print("4. Start chatting!")
    else:
        print("❌ Some tests failed. Please fix the issues above before running the agent.")
        print("\n🔧 Common fixes:")
        print("1. Install missing packages: pip install -r requirements.txt")
        print("2. Set environment variables: export ASI1_API_KEY='your-key'")
        print("3. Check Python version (3.8+ required)")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 