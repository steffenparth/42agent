import json
import os
from typing import Any, Optional
import requests
import logging

# Configure logging
logger = logging.getLogger(__name__)

# ASI API Configuration
ASI_API_KEY = "sk_cf9e5b12404f4217a875461712a082374a4f34765b3f409e97f4cbdbe1cf01a6"

HEADERS = {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
    'Authorization': f'Bearer {ASI_API_KEY}'
}

URL = "https://api.asi1.ai/v1/chat/completions"
MODEL = "asi1-mini"  # Using ASI One Mini model (available with this API key)

# Default configuration
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1000
DEFAULT_STREAM = False

class LLMConfig:
    """Configuration class for LLM settings"""
    
    def __init__(self):
        self.model = MODEL
        self.temperature = DEFAULT_TEMPERATURE
        self.max_tokens = DEFAULT_MAX_TOKENS
        self.stream = DEFAULT_STREAM
        self.api_key = ASI_API_KEY
        self.base_url = URL

# Global configuration instance
llm_config = LLMConfig()

async def get_completion(
    context: str = "", 
    prompt: str = "", 
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    model: Optional[str] = None
) -> dict:
    """
    Get completion from ASI LLM with enhanced error handling and configuration
    
    Args:
        context: Conversation context
        prompt: User prompt
        temperature: Sampling temperature (0.0 to 2.0)
        max_tokens: Maximum tokens to generate
        model: Model to use
    
    Returns:
        dict: LLM response or error information
    """
    
    try:
        # Use provided parameters or defaults
        temp = temperature if temperature is not None else llm_config.temperature
        tokens = max_tokens if max_tokens is not None else llm_config.max_tokens
        model_name = model if model else llm_config.model
        
        # Prepare messages
        messages = []
        
        # Add system context if provided
        if context.strip():
            messages.append({
                "role": "system",
                "content": context
            })
        
        # Add user message
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temp,
            "stream": llm_config.stream,
            "max_tokens": tokens
        }
        
        logger.info(f"🤖 Sending request to ASI LLM (model: {model_name})")
        logger.debug(f"📝 Prompt: {prompt[:100]}...")
        
        # Make API request
        response = requests.post(
            URL, 
            headers=HEADERS, 
            json=payload,
            timeout=30  # 30 second timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            logger.info("✅ LLM response received successfully")
            return result
        else:
            error_msg = f"API request failed with status {response.status_code}: {response.text}"
            logger.error(f"❌ {error_msg}")
            return {
                "error": error_msg,
                "status_code": response.status_code,
                "choices": [{"message": {"content": f"Error: {error_msg}"}}]
            }
            
    except requests.exceptions.Timeout:
        error_msg = "Request timed out"
        logger.error(f"⏰ {error_msg}")
        return {
            "error": error_msg,
            "choices": [{"message": {"content": "Sorry, the request timed out. Please try again."}}]
        }
        
    except requests.exceptions.RequestException as e:
        error_msg = f"Network error: {str(e)}"
        logger.error(f"🌐 {error_msg}")
        return {
            "error": error_msg,
            "choices": [{"message": {"content": "Sorry, there was a network error. Please check your connection."}}]
        }
        
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(f"💥 {error_msg}")
        return {
            "error": error_msg,
            "choices": [{"message": {"content": "Sorry, an unexpected error occurred."}}]
        }

async def get_completion_simple(prompt: str) -> str:
    """
    Simple wrapper for get_completion that returns just the text response
    
    Args:
        prompt: User prompt
        
    Returns:
        str: LLM response text or error message
    """
    try:
        result = await get_completion(prompt=prompt)
        
        if "error" in result:
            return f"Error: {result['error']}"
        
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        else:
            return "No response generated"
            
    except Exception as e:
        logger.error(f"Error in simple completion: {e}")
        return f"Error: {str(e)}"

def update_config(
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None
):
    """
    Update LLM configuration
    
    Args:
        temperature: New temperature setting
        max_tokens: New max tokens setting
        model: New model name
        api_key: New API key
    """
    global llm_config, HEADERS
    
    if temperature is not None:
        llm_config.temperature = temperature
        
    if max_tokens is not None:
        llm_config.max_tokens = max_tokens
        
    if model is not None:
        llm_config.model = model
        
    if api_key is not None:
        llm_config.api_key = api_key
        HEADERS['Authorization'] = f'Bearer {api_key}'
    
    logger.info("⚙️ LLM configuration updated")

def get_config() -> dict:
    """Get current LLM configuration"""
    return {
        "model": llm_config.model,
        "temperature": llm_config.temperature,
        "max_tokens": llm_config.max_tokens,
        "stream": llm_config.stream,
        "base_url": llm_config.base_url
    }

# Test function
async def test_llm():
    """Test the LLM integration"""
    print("🧪 Testing LLM integration...")
    
    test_prompt = "Hello! Can you tell me a short joke?"
    result = await get_completion(prompt=test_prompt)
    
    if "error" in result:
        print(f"❌ Test failed: {result['error']}")
        return False
    else:
        response = result["choices"][0]["message"]["content"]
        print(f"✅ Test successful!")
        print(f"🤖 Response: {response}")
        return True

if __name__ == "__main__":
    import asyncio
    
    print("🚀 ASI LLM Integration")
    print(f"📡 API URL: {URL}")
    print(f"🤖 Model: {MODEL}")
    print(f"🔑 API Key: {ASI_API_KEY[:20]}...")
    
    # Run test
    asyncio.run(test_llm())