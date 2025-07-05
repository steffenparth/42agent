#!/usr/bin/env python3
"""
Setup Script for ASI:One Agent

This script helps set up the ASI:One agent environment and dependencies.
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"   stdout: {e.stdout}")
        if e.stderr:
            print(f"   stderr: {e.stderr}")
        return False

def check_python_version() -> bool:
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_dependencies() -> bool:
    """Install required dependencies."""
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found")
        return False
    
    return run_command("pip install -r requirements.txt", "Installing dependencies")

def setup_environment() -> bool:
    """Set up environment variables."""
    print("🔧 Setting up environment variables...")
    
    # Check if ASI1_API_KEY is already set
    if os.getenv("ASI1_API_KEY"):
        print("✅ ASI1_API_KEY is already set")
        return True
    
    # Prompt for API key
    print("📝 Please enter your ASI:One API key:")
    api_key = input("ASI1_API_KEY: ").strip()
    
    if not api_key:
        print("❌ No API key provided")
        return False
    
    # Set environment variable for current session
    os.environ["ASI1_API_KEY"] = api_key
    print("✅ ASI1_API_KEY set for current session")
    
    # Create .env file for future sessions
    env_file = Path(".env")
    if not env_file.exists():
        with open(env_file, "w") as f:
            f.write(f"ASI1_API_KEY={api_key}\n")
        print("✅ Created .env file for future sessions")
    
    return True

def run_tests() -> bool:
    """Run the test setup script."""
    if not os.path.exists("test_setup.py"):
        print("❌ test_setup.py not found")
        return False
    
    return run_command("python test_setup.py", "Running setup tests")

def main():
    """Main setup function."""
    print("🚀 ASI:One Agent Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Setup environment
    if not setup_environment():
        return False
    
    # Run tests
    if not run_tests():
        return False
    
    print("\n🎉 Setup completed successfully!")
    print("\n🚀 Next steps:")
    print("1. Start the agent: python ai_agent.py")
    print("2. In another terminal, start the client: python client.py")
    print("3. Enter the agent address when prompted")
    print("4. Start chatting with your ASI:One agent!")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 