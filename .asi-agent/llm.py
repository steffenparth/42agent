"""
LLM Module for ASI:One Integration

This module provides the interface to ASI:One's LLM API for generating completions.
It handles authentication, API calls, and response processing.
"""

import json
import os
from typing import Any, Dict, Optional
import requests
from datetime import datetime

# ASI:One API Configuration
HEADERS = {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
    'Authorization': f'bearer {os.getenv("ASI1_API_KEY", "")}'
}

URL = "https://api.asi1.ai/v1/chat/completions"
MODEL = "asi1-mini"

class ASIOneError(Exception):
    """Custom exception for ASI:One API errors"""
    pass

async def get_completion(context: str, prompt: str) -> Dict[str, Any]:
    """
    Get a completion from ASI:One's LLM API.
    
    Args:
        context: Additional context for the prompt
        prompt: The main prompt to send to the LLM
        
    Returns:
        Dict containing the API response
        
    Raises:
        ASIOneError: If the API call fails or returns an error
    """
    # Validate API key
    if not os.getenv("ASI1_API_KEY"):
        raise ASIOneError("ASI1_API_KEY environment variable not set")
    
    # Prepare the payload
    payload = json.dumps({
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": context + " " + prompt
            }
        ],
        "temperature": 0,
        "stream": False,
        "max_tokens": 0
    })
    
    try:
        # Make the API request
        response = requests.post(URL, headers=HEADERS, data=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        
        # Check for API errors in the response
        if "error" in result:
            raise ASIOneError(f"ASI:One API error: {result['error']}")
        
        return result
        
    except requests.exceptions.RequestException as e:
        raise ASIOneError(f"Request failed: {str(e)}")
    except json.JSONDecodeError as e:
        raise ASIOneError(f"Invalid JSON response: {str(e)}")
    except Exception as e:
        raise ASIOneError(f"Unexpected error: {str(e)}")

def extract_completion_text(response: Dict[str, Any]) -> str:
    """
    Extract the completion text from the ASI:One API response.
    
    Args:
        response: The API response dictionary
        
    Returns:
        The completion text string
        
    Raises:
        ASIOneError: If the response format is invalid
    """
    try:
        return response["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise ASIOneError(f"Invalid response format: {str(e)}")

async def get_completion_text(context: str, prompt: str) -> str:
    """
    Get completion text directly from ASI:One API.
    
    Args:
        context: Additional context for the prompt
        prompt: The main prompt to send to the LLM
        
    Returns:
        The completion text string
    """
    response = await get_completion(context, prompt)
    return extract_completion_text(response) 