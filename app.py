"""
AI Bias Research Tool - Main Application
Created: December 13, 2024
Last Updated: December 15, 2024

FIXES:
- December 14, 2024: Fixed OpenAI client initialization for v1.0+ API
- December 14, 2024: Updated Gemini model naming attempts
- December 14, 2024: Switched to direct REST API calls for Gemini
- December 15, 2024: FIXED! Updated to use Gemini 2.0/2.5 models (1.5 models deprecated)
- December 15, 2024: Added system prompts to force number-first responses with 3 decimal places
- December 15, 2024: Changed rating storage from INTEGER to REAL for decimal precision
- December 15, 2024: Simplified rating extraction (now just parses first number in response)
- December 15, 2024: FIXED ANTHROPIC! Updated Claude model from claude-3-5-sonnet-20241022 
                      to claude-sonnet-4-20250514 (Claude Sonnet 4) - old model was deprecated
- December 15, 2024: ADDED DEEPSEEK! Integrated DeepSeek AI from China (deepseek-chat model)
- December 15, 2024: FIXED COHERE! Updated from deprecated command-r-plus to command-a-03-2025
                      (Command A - Cohere's most performant model)
- December 15, 2024: FIXED GROQ/LLAMA! Updated from deprecated llama-3.1-70b-versatile 
                      to llama-3.3-70b-versatile (Meta Llama 3.3 with quality improvements)
- December 15, 2024: REPLACED QWEN WITH AI21! Swapped Alibaba Qwen Plus for AI21 Jamba-1.5-Large
                      (Israeli AI with 256K context, hybrid Mamba-Transformer architecture)
- December 15, 2024: ADDED GROK! Integrated xAI Grok-Beta as 10th AI system
                      (Elon Musk's AI with real-time X data access capabilities)

This application queries multiple AI systems with the same question to detect bias patterns.
Designed for research purposes to cross-validate AI responses.

Author: Jim (Hyperiongate)
Purpose: Discover if there's "any there there" in AI bias detection

AI SYSTEMS INTEGRATED (10 total):
- OpenAI GPT-4 (USA) - Proprietary
- OpenAI GPT-3.5-Turbo (USA) - Proprietary
- Google Gemini-2.0-Flash (USA) - Proprietary
- Anthropic Claude-Sonnet-4 (USA) - Proprietary
- Mistral Large-2 (France) - Proprietary
- DeepSeek Chat (China) - Proprietary
- Cohere Command A (Canada) - Proprietary - UPDATED TO LATEST MODEL!
- Meta Llama 3.3 70B via Groq (USA) - OPEN SOURCE - UPDATED TO LATEST MODEL!
- AI21 Jamba-1.5-Large (Israel) - Proprietary - NEW! Replaces Qwen
- xAI Grok-Beta (USA) - Proprietary - NEW! 10th AI system!
"""

from flask import Flask, render_template, request, jsonify
import os
import sqlite3
from datetime import datetime
from openai import OpenAI
from ai21 import AI21Client
from ai21.models.chat import ChatMessage as AI21ChatMessage
import requests
import json
import time
import re

app = Flask(__name__)

# Configure API keys from environment variables
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')
MISTRAL_API_KEY = os.environ.get('MISTRAL_API_KEY')
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY')
COHERE_API_KEY = os.environ.get('COHERE_API_KEY')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
AI21_API_KEY = os.environ.get('AI21_API_KEY')
XAI_API_KEY = os.environ.get('XAI_API_KEY')

# Initialize OpenAI client
openai_client = None
if OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize DeepSeek client (uses OpenAI-compatible API)
deepseek_client = None
if DEEPSEEK_API_KEY:
    deepseek_client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com"
    )

# Initialize Groq client (uses OpenAI-compatible API)
groq_client = None
if GROQ_API_KEY:
    groq_client = OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1"
    )

# Initialize AI21 client
ai21_client = None
if AI21_API_KEY:
    ai21_client = AI21Client(api_key=AI21_API_KEY)

# Initialize xAI Grok client (uses OpenAI-compatible API)
xai_client = None
if XAI_API_KEY:
    xai_client = OpenAI(
        api_key=XAI_API_KEY,
        base_url="https://api.x.ai/v1"
    )

# System prompt to ensure consistent, parseable responses
RATING_SYSTEM_PROMPT = """You are participating in a research study on AI responses. When asked to rate something on a numerical scale, you MUST follow these rules:

1. Start your response with ONLY the numerical rating on the first line
2. Use up to 3 decimal places for precision (e.g., 7.250, 8.125, 6.875)
3. Then provide your explanation on subsequent lines

Example format:
7.250

Your explanation goes here...

This format is critical for data collection. Always provide a specific number, never a range."""

# Database setup
DATABASE = 'bias_research.db'

def get_db():
    """Get database connection"""
    db = sqlite3.connect(DATABASE)
    db.row_factory = sqlite3.Row
    return db

def init_db():
    """Initialize database with schema.
    
    Note: extracted_rating is REAL to support decimal values up to 3 decimal places.
    """
    db = get_db()
    db.execute('''
        CREATE TABLE IF NOT EXISTS queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    db.execute('''
        CREATE TABLE IF NOT EXISTS responses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_id INTEGER NOT NULL,
            ai_system TEXT NOT NULL,
            model TEXT NOT NULL,
            raw_response TEXT NOT NULL,
            extracted_rating REAL,
            response_time REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (query_id) REFERENCES queries(id)
        )
    ''')
    db.commit()
    db.close()

# Initialize database on startup
init_db()

def query_openai_gpt4(question):
    """Query OpenAI GPT-4 with system prompt for structured responses."""
    if not openai_client:
        return {
            'success': False,
            'error': 'OpenAI API key not configured',
            'system': 'OpenAI',
            'model': 'GPT-4'
        }
    
    try:
        start_time = time.time()
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": RATING_SYSTEM_PROMPT},
                {"role": "user", "content": question}
            ],
            temperature=0.7,
            max_tokens=500
        )
        response_time = time.time() - start_time
        
        raw_response = response.choices[0].message.content
        
        return {
            'success': True,
            'system': 'OpenAI',
            'model': 'GPT-4',
            'raw_response': raw_response,
            'response_time': response_time
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'system': 'OpenAI',
            'model': 'GPT-4'
        }

def query_openai_gpt35(question):
    """Query OpenAI GPT-3.5 Turbo with system prompt for structured responses."""
    if not openai_client:
        return {
            'success': False,
            'error': 'OpenAI API key not configured',
            'system': 'OpenAI',
            'model': 'GPT-3.5-Turbo'
        }
    
    try:
        start_time = time.time()
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": RATING_SYSTEM_PROMPT},
                {"role": "user", "content": question}
            ],
            temperature=0.7,
            max_tokens=500
        )
        response_time = time.time() - start_time
        
        raw_response = response.choices[0].message.content
        
        return {
            'success': True,
            'system': 'OpenAI',
            'model': 'GPT-3.5-Turbo',
            'raw_response': raw_response,
            'response_time': response_time
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'system': 'OpenAI',
            'model': 'GPT-3.5-Turbo'
        }

def query_google_gemini(question):
    """Query Google Gemini with system prompt for structured responses.
    
    Uses Gemini 2.0 Flash via v1beta endpoint.
    """
    if not GOOGLE_API_KEY:
        return {
            'success': False,
            'error': 'Google API key not configured',
            'system': 'Google',
            'model': 'Gemini'
        }
    
    try:
        start_time = time.time()
        
        model_name = 'gemini-2.0-flash'
        api_version = 'v1beta'
        
        url = f"https://generativelanguage.googleapis.com/{api_version}/models/{model_name}:generateContent"
        
        headers = {
            'Content-Type': 'application/json'
        }
        
        payload = {
            'systemInstruction': {
                'parts': [{
                    'text': RATING_SYSTEM_PROMPT
                }]
            },
            'contents': [{
                'parts': [{
                    'text': question
                }]
            }],
            'generationConfig': {
                'temperature': 0.7,
                'maxOutputTokens': 500
            }
        }
        
        response = requests.post(
            url,
            headers=headers,
            params={'key': GOOGLE_API_KEY},
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            response_time = time.time() - start_time
            data = response.json()
            
            if 'candidates' in data and len(data['candidates']) > 0:
                candidate = data['candidates'][0]
                if 'content' in candidate and 'parts' in candidate['content']:
                    raw_response = candidate['content']['parts'][0].get('text', '')
                    
                    return {
                        'success': True,
                        'system': 'Google',
                        'model': 'Gemini-2.0-Flash',
                        'raw_response': raw_response,
                        'response_time': response_time
                    }
            
            return {
                'success': False,
                'error': 'Unexpected response format',
                'system': 'Google',
                'model': 'Gemini-2.0-Flash'
            }
        else:
            try:
                error_data = response.json()
                error_msg = error_data.get('error', {}).get('message', f"HTTP {response.status_code}")
            except:
                error_msg = f"HTTP {response.status_code}: {response.text[:200]}"
            
            return {
                'success': False,
                'error': error_msg,
                'system': 'Google',
                'model': 'Gemini-2.0-Flash'
            }
        
    except requests.exceptions.Timeout:
        return {
            'success': False,
            'error': 'Request timed out after 30 seconds',
            'system': 'Google',
            'model': 'Gemini-2.0-Flash'
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'system': 'Google',
            'model': 'Gemini-2.0-Flash'
        }

def query_anthropic_claude(question):
    """Query Anthropic Claude with system prompt for structured responses.
    
    Uses Claude Sonnet 4 via direct REST API calls.
    Model: claude-sonnet-4-20250514
    """
    if not ANTHROPIC_API_KEY:
        return {
            'success': False,
            'error': 'Anthropic API key not configured',
            'system': 'Anthropic',
            'model': 'Claude'
        }
    
    try:
        start_time = time.time()
        
        url = "https://api.anthropic.com/v1/messages"
        
        headers = {
            'Content-Type': 'application/json',
            'x-api-key': ANTHROPIC_API_KEY,
            'anthropic-version': '2023-06-01'
        }
        
        payload = {
            'model': 'claude-sonnet-4-20250514',
            'max_tokens': 500,
            'system': RATING_SYSTEM_PROMPT,
            'messages': [
                {
                    'role': 'user',
                    'content': question
                }
            ]
        }
        
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            response_time = time.time() - start_time
            data = response.json()
            
            if 'content' in data and len(data['content']) > 0:
                raw_response = data['content'][0].get('text', '')
                
                return {
                    'success': True,
                    'system': 'Anthropic',
                    'model': 'Claude-Sonnet-4',
                    'raw_response': raw_response,
                    'response_time': response_time
                }
            
            return {
                'success': False,
                'error': 'Unexpected response format from Anthropic API',
                'system': 'Anthropic',
                'model': 'Claude-Sonnet-4'
            }
        else:
            try:
                error_data = response.json()
                error_type = error_data.get('error', {}).get('type', 'unknown')
                error_msg = error_data.get('error', {}).get('message', f"HTTP {response.status_code}")
                full_error = f"{error_type}: {error_msg}"
            except:
                full_error = f"HTTP {response.status_code}: {response.text[:200]}"
            
            return {
                'success': False,
                'error': full_error,
                'system': 'Anthropic',
                'model': 'Claude-Sonnet-4'
            }
        
    except requests.exceptions.Timeout:
        return {
            'success': False,
            'error': 'Request timed out after 30 seconds',
            'system': 'Anthropic',
            'model': 'Claude-Sonnet-4'
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'system': 'Anthropic',
            'model': 'Claude-Sonnet-4'
        }

def query_mistral_large(question):
    """Query Mistral Large with system prompt for structured responses.
    
    Uses Mistral Large 2 (mistral-large-latest) via REST API.
    Provides European (French) perspective on AI responses.
    """
    if not MISTRAL_API_KEY:
        return {
            'success': False,
            'error': 'Mistral API key not configured',
            'system': 'Mistral',
            'model': 'Large-2'
        }
    
    try:
        start_time = time.time()
        
        url = "https://api.mistral.ai/v1/chat/completions"
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {MISTRAL_API_KEY}'
        }
        
        payload = {
            'model': 'mistral-large-latest',
            'messages': [
                {
                    'role': 'system',
                    'content': RATING_SYSTEM_PROMPT
                },
                {
                    'role': 'user',
                    'content': question
                }
            ],
            'temperature': 0.7,
            'max_tokens': 500
        }
        
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            response_time = time.time() - start_time
            data = response.json()
            
            if 'choices' in data and len(data['choices']) > 0:
                raw_response = data['choices'][0].get('message', {}).get('content', '')
                
                return {
                    'success': True,
                    'system': 'Mistral',
                    'model': 'Large-2',
                    'raw_response': raw_response,
                    'response_time': response_time
                }
            
            return {
                'success': False,
                'error': 'Unexpected response format from Mistral API',
                'system': 'Mistral',
                'model': 'Large-2'
            }
        else:
            try:
                error_data = response.json()
                error_msg = error_data.get('message', f"HTTP {response.status_code}")
            except:
                error_msg = f"HTTP {response.status_code}: {response.text[:200]}"
            
            return {
                'success': False,
                'error': error_msg,
                'system': 'Mistral',
                'model': 'Large-2'
            }
        
    except requests.exceptions.Timeout:
        return {
            'success': False,
            'error': 'Request timed out after 30 seconds',
            'system': 'Mistral',
            'model': 'Large-2'
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'system': 'Mistral',
            'model': 'Large-2'
        }

def query_deepseek_chat(question):
    """Query DeepSeek Chat with system prompt for structured responses.
    
    Uses DeepSeek V3.2 (deepseek-chat) via OpenAI-compatible API.
    Provides Chinese AI perspective on responses.
    
    API Endpoint: https://api.deepseek.com
    Model: deepseek-chat
    """
    if not deepseek_client:
        return {
            'success': False,
            'error': 'DeepSeek API key not configured',
            'system': 'DeepSeek',
            'model': 'Chat'
        }
    
    try:
        start_time = time.time()
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": RATING_SYSTEM_PROMPT},
                {"role": "user", "content": question}
            ],
            temperature=0.7,
            max_tokens=500
        )
        response_time = time.time() - start_time
        
        raw_response = response.choices[0].message.content
        
        return {
            'success': True,
            'system': 'DeepSeek',
            'model': 'Chat-V3',
            'raw_response': raw_response,
            'response_time': response_time
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'system': 'DeepSeek',
            'model': 'Chat-V3'
        }

def query_cohere_command(question):
    """Query Cohere Command A with system prompt for structured responses.
    
    Uses Cohere Command A (command-a-03-2025) via REST API v2.
    This is Cohere's most performant model, replacing the deprecated Command R+.
    
    UPDATED December 15, 2024: Changed from deprecated 'command-r-plus' to 'command-a-03-2025'
    Command A is Cohere's flagship model with 111B parameters, 256K context, and best performance.
    
    Provides Canadian AI perspective on responses.
    
    API Endpoint: https://api.cohere.com/v2/chat
    Model: command-a-03-2025
    """
    if not COHERE_API_KEY:
        return {
            'success': False,
            'error': 'Cohere API key not configured',
            'system': 'Cohere',
            'model': 'Command-A'
        }
    
    try:
        start_time = time.time()
        
        url = "https://api.cohere.com/v2/chat"
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {COHERE_API_KEY}'
        }
        
        payload = {
            'model': 'command-a-03-2025',
            'messages': [
                {
                    'role': 'system',
                    'content': RATING_SYSTEM_PROMPT
                },
                {
                    'role': 'user',
                    'content': question
                }
            ],
            'temperature': 0.7,
            'max_tokens': 500
        }
        
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            response_time = time.time() - start_time
            data = response.json()
            
            if 'message' in data and 'content' in data['message']:
                content = data['message']['content']
                if isinstance(content, list) and len(content) > 0:
                    raw_response = content[0].get('text', '')
                else:
                    raw_response = str(content)
                
                return {
                    'success': True,
                    'system': 'Cohere',
                    'model': 'Command-A',
                    'raw_response': raw_response,
                    'response_time': response_time
                }
            
            return {
                'success': False,
                'error': 'Unexpected response format from Cohere API',
                'system': 'Cohere',
                'model': 'Command-A'
            }
        else:
            try:
                error_data = response.json()
                error_msg = error_data.get('message', f"HTTP {response.status_code}")
            except:
                error_msg = f"HTTP {response.status_code}: {response.text[:200]}"
            
            return {
                'success': False,
                'error': error_msg,
                'system': 'Cohere',
                'model': 'Command-A'
            }
        
    except requests.exceptions.Timeout:
        return {
            'success': False,
            'error': 'Request timed out after 30 seconds',
            'system': 'Cohere',
            'model': 'Command-A'
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'system': 'Cohere',
            'model': 'Command-A'
        }

def query_groq_llama(question):
    """Query Meta Llama 3.3 70B via Groq with system prompt for structured responses.
    
    Uses Llama 3.3 70B (llama-3.3-70b-versatile) via Groq's ultra-fast LPU inference.
    This is an OPEN SOURCE model, unlike all other proprietary models.
    
    UPDATED December 15, 2024: Changed from deprecated 'llama-3.1-70b-versatile' 
    to 'llama-3.3-70b-versatile' with significant quality improvements.
    
    API Endpoint: https://api.groq.com/openai/v1
    Model: llama-3.3-70b-versatile
    """
    if not groq_client:
        return {
            'success': False,
            'error': 'Groq API key not configured',
            'system': 'Meta',
            'model': 'Llama-3.3-70B'
        }
    
    try:
        start_time = time.time()
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": RATING_SYSTEM_PROMPT},
                {"role": "user", "content": question}
            ],
            temperature=0.7,
            max_tokens=500
        )
        response_time = time.time() - start_time
        
        raw_response = response.choices[0].message.content
        
        return {
            'success': True,
            'system': 'Meta (via Groq)',
            'model': 'Llama-3.3-70B',
            'raw_response': raw_response,
            'response_time': response_time
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'system': 'Meta (via Groq)',
            'model': 'Llama-3.3-70B'
        }

def query_ai21_jamba(question):
    """Query AI21 Jamba-1.5-Large with system prompt for structured responses.
    
    Uses Jamba-1.5-Large via AI21 Python SDK.
    Provides Israeli AI perspective on responses.
    
    AI21 Labs' Jamba is a hybrid SSM-Transformer (Mamba-Transformer) model with:
    - 256K context window (one of the largest available)
    - Efficient processing of long documents
    - Excellent instruction-following capabilities
    
    API Endpoint: https://api.ai21.com/studio/v1/chat/completions
    Model: jamba-1.5-large
    
    ADDED December 15, 2024 to replace Alibaba Qwen Plus.
    Provides Middle Eastern AI perspective (Israel) for geographic diversity.
    """
    if not ai21_client:
        return {
            'success': False,
            'error': 'AI21 API key not configured. Get your API key from: https://studio.ai21.com/account/api-key',
            'system': 'AI21',
            'model': 'Jamba-1.5-Large'
        }
    
    try:
        start_time = time.time()
        
        messages = [
            AI21ChatMessage(role="system", content=RATING_SYSTEM_PROMPT),
            AI21ChatMessage(role="user", content=question)
        ]
        
        response = ai21_client.chat.completions.create(
            model="jamba-1.5-large",
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )
        
        response_time = time.time() - start_time
        
        raw_response = response.choices[0].message.content
        
        return {
            'success': True,
            'system': 'AI21',
            'model': 'Jamba-1.5-Large',
            'raw_response': raw_response,
            'response_time': response_time
        }
    except Exception as e:
        error_msg = str(e)
        # Add helpful context for common errors
        if 'Unauthorized' in error_msg or 'Invalid API key' in error_msg:
            error_msg += ' - Verify your AI21_API_KEY in Render environment variables'
        elif 'Model not found' in error_msg:
            error_msg += ' - Model jamba-1.5-large may not be available'
        
        return {
            'success': False,
            'error': error_msg,
            'system': 'AI21',
            'model': 'Jamba-1.5-Large'
        }

def query_xai_grok(question):
    """Query xAI Grok-Beta with system prompt for structured responses.
    
    Uses Grok-Beta via OpenAI-compatible API.
    Provides xAI (Elon Musk's AI) perspective on responses.
    
    xAI's Grok is designed to be "maximally truth-seeking" and has access to real-time
    X (Twitter) data, giving it unique current events capabilities.
    
    API Endpoint: https://api.x.ai/v1
    Model: grok-beta
    
    ADDED December 15, 2024 as 10th AI system.
    Provides additional USA perspective with unique real-time data access.
    """
    if not xai_client:
        return {
            'success': False,
            'error': 'xAI API key not configured. Get your API key from: https://console.x.ai',
            'system': 'xAI',
            'model': 'Grok-Beta'
        }
    
    try:
        start_time = time.time()
        response = xai_client.chat.completions.create(
            model="grok-beta",
            messages=[
                {"role": "system", "content": RATING_SYSTEM_PROMPT},
                {"role": "user", "content": question}
            ],
            temperature=0.7,
            max_tokens=500
        )
        response_time = time.time() - start_time
        
        raw_response = response.choices[0].message.content
        
        return {
            'success': True,
            'system': 'xAI',
            'model': 'Grok-Beta',
            'raw_response': raw_response,
            'response_time': response_time
        }
    except Exception as e:
        error_msg = str(e)
        # Add helpful context for common errors
        if 'Unauthorized' in error_msg or 'Invalid API key' in error_msg:
            error_msg += ' - Verify your XAI_API_KEY in Render environment variables'
        elif 'Model not found' in error_msg:
            error_msg += ' - Model grok-beta may not be available. Try grok-4 if available.'
        
        return {
            'success': False,
            'error': error_msg,
            'system': 'xAI',
            'model': 'Grok-Beta'
        }

def extract_rating(text):
    """
    Extract numerical rating from the response.
    
    Since we're using system prompts that instruct the AI to put the number
    on the first line, this function looks for the first number in the response.
    
    Returns float with up to 3 decimal places, or None if no rating found.
    """
    if not text:
        return None
    
    first_line = text.strip().split('\n')[0].strip()
    
    match = re.search(r'^(\d+(?:\.\d+)?)', first_line)
    
    if match:
        try:
            rating = float(match.group(1))
            if 0 <= rating <= 10:
                return round(rating, 3)
        except ValueError:
            pass
    
    match = re.search(r'(\d+(?:\.\d+)?)\s*/\s*10', text)
    if match:
        try:
            rating = float(match.group(1))
            if 0 <= rating <= 10:
                return round(rating, 3)
        except ValueError:
            pass
    
    return None

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query_ais():
    """Query all 10 AI systems with the same question"""
    data = request.json
    question = data.get('question', '').strip()
    
    if not question:
        return jsonify({'error': 'Question is required'}), 400
    
    db = get_db()
    cursor = db.execute('INSERT INTO queries (question) VALUES (?)', (question,))
    query_id = cursor.lastrowid
    db.commit()
    
    results = []
    
    # OpenAI GPT-4
    gpt4_result = query_openai_gpt4(question)
    if gpt4_result['success']:
        extracted_rating = extract_rating(gpt4_result['raw_response'])
        db.execute('''
            INSERT INTO responses 
            (query_id, ai_system, model, raw_response, extracted_rating, response_time)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (query_id, gpt4_result['system'], gpt4_result['model'], 
              gpt4_result['raw_response'], extracted_rating, gpt4_result['response_time']))
        gpt4_result['extracted_rating'] = extracted_rating
    results.append(gpt4_result)
    
    # OpenAI GPT-3.5
    gpt35_result = query_openai_gpt35(question)
    if gpt35_result['success']:
        extracted_rating = extract_rating(gpt35_result['raw_response'])
        db.execute('''
            INSERT INTO responses 
            (query_id, ai_system, model, raw_response, extracted_rating, response_time)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (query_id, gpt35_result['system'], gpt35_result['model'], 
              gpt35_result['raw_response'], extracted_rating, gpt35_result['response_time']))
        gpt35_result['extracted_rating'] = extracted_rating
    results.append(gpt35_result)
    
    # Google Gemini
    gemini_result = query_google_gemini(question)
    if gemini_result['success']:
        extracted_rating = extract_rating(gemini_result['raw_response'])
        db.execute('''
            INSERT INTO responses 
            (query_id, ai_system, model, raw_response, extracted_rating, response_time)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (query_id, gemini_result['system'], gemini_result['model'], 
              gemini_result['raw_response'], extracted_rating, gemini_result['response_time']))
        gemini_result['extracted_rating'] = extracted_rating
    results.append(gemini_result)
    
    # Anthropic Claude
    claude_result = query_anthropic_claude(question)
    if claude_result['success']:
        extracted_rating = extract_rating(claude_result['raw_response'])
        db.execute('''
            INSERT INTO responses 
            (query_id, ai_system, model, raw_response, extracted_rating, response_time)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (query_id, claude_result['system'], claude_result['model'], 
              claude_result['raw_response'], extracted_rating, claude_result['response_time']))
        claude_result['extracted_rating'] = extracted_rating
    results.append(claude_result)
    
    # Mistral Large
    mistral_result = query_mistral_large(question)
    if mistral_result['success']:
        extracted_rating = extract_rating(mistral_result['raw_response'])
        db.execute('''
            INSERT INTO responses 
            (query_id, ai_system, model, raw_response, extracted_rating, response_time)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (query_id, mistral_result['system'], mistral_result['model'], 
              mistral_result['raw_response'], extracted_rating, mistral_result['response_time']))
        mistral_result['extracted_rating'] = extracted_rating
    results.append(mistral_result)
    
    # DeepSeek Chat (China)
    deepseek_result = query_deepseek_chat(question)
    if deepseek_result['success']:
        extracted_rating = extract_rating(deepseek_result['raw_response'])
        db.execute('''
            INSERT INTO responses 
            (query_id, ai_system, model, raw_response, extracted_rating, response_time)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (query_id, deepseek_result['system'], deepseek_result['model'], 
              deepseek_result['raw_response'], extracted_rating, deepseek_result['response_time']))
        deepseek_result['extracted_rating'] = extracted_rating
    results.append(deepseek_result)
    
    # Cohere Command A (Canada) - UPDATED MODEL!
    cohere_result = query_cohere_command(question)
    if cohere_result['success']:
        extracted_rating = extract_rating(cohere_result['raw_response'])
        db.execute('''
            INSERT INTO responses 
            (query_id, ai_system, model, raw_response, extracted_rating, response_time)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (query_id, cohere_result['system'], cohere_result['model'], 
              cohere_result['raw_response'], extracted_rating, cohere_result['response_time']))
        cohere_result['extracted_rating'] = extracted_rating
    results.append(cohere_result)
    
    # Meta Llama 3.3 70B via Groq (Open Source) - UPDATED MODEL!
    llama_result = query_groq_llama(question)
    if llama_result['success']:
        extracted_rating = extract_rating(llama_result['raw_response'])
        db.execute('''
            INSERT INTO responses 
            (query_id, ai_system, model, raw_response, extracted_rating, response_time)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (query_id, llama_result['system'], llama_result['model'], 
              llama_result['raw_response'], extracted_rating, llama_result['response_time']))
        llama_result['extracted_rating'] = extracted_rating
    results.append(llama_result)
    
    # AI21 Jamba-1.5-Large (Israel) - NEW!
    jamba_result = query_ai21_jamba(question)
    if jamba_result['success']:
        extracted_rating = extract_rating(jamba_result['raw_response'])
        db.execute('''
            INSERT INTO responses 
            (query_id, ai_system, model, raw_response, extracted_rating, response_time)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (query_id, jamba_result['system'], jamba_result['model'], 
              jamba_result['raw_response'], extracted_rating, jamba_result['response_time']))
        jamba_result['extracted_rating'] = extracted_rating
    results.append(jamba_result)
    
    # xAI Grok-Beta (USA) - NEW! 10th AI system!
    grok_result = query_xai_grok(question)
    if grok_result['success']:
        extracted_rating = extract_rating(grok_result['raw_response'])
        db.execute('''
            INSERT INTO responses 
            (query_id, ai_system, model, raw_response, extracted_rating, response_time)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (query_id, grok_result['system'], grok_result['model'], 
              grok_result['raw_response'], extracted_rating, grok_result['response_time']))
        grok_result['extracted_rating'] = extracted_rating
    results.append(grok_result)
    
    db.commit()
    db.close()
    
    return jsonify({
        'query_id': query_id,
        'question': question,
        'results': results,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/history')
def get_history():
    """Get query history"""
    db = get_db()
    queries = db.execute('''
        SELECT q.id, q.question, q.timestamp,
               COUNT(r.id) as response_count
        FROM queries q
        LEFT JOIN responses r ON q.id = r.query_id
        GROUP BY q.id
        ORDER BY q.timestamp DESC
        LIMIT 50
    ''').fetchall()
    
    history = []
    for query in queries:
        history.append({
            'id': query['id'],
            'question': query['question'],
            'timestamp': query['timestamp'],
            'response_count': query['response_count']
        })
    
    db.close()
    return jsonify(history)

@app.route('/query/<int:query_id>')
def get_query_details(query_id):
    """Get details of a specific query and its responses"""
    db = get_db()
    
    query = db.execute('SELECT * FROM queries WHERE id = ?', (query_id,)).fetchone()
    if not query:
        return jsonify({'error': 'Query not found'}), 404
    
    responses = db.execute('''
        SELECT * FROM responses WHERE query_id = ? ORDER BY id
    ''', (query_id,)).fetchall()
    
    result = {
        'id': query['id'],
        'question': query['question'],
        'timestamp': query['timestamp'],
        'responses': []
    }
    
    for response in responses:
        result['responses'].append({
            'id': response['id'],
            'ai_system': response['ai_system'],
            'model': response['model'],
            'raw_response': response['raw_response'],
            'extracted_rating': response['extracted_rating'],
            'response_time': response['response_time'],
            'timestamp': response['timestamp']
        })
    
    db.close()
    return jsonify(result)

@app.route('/reset', methods=['POST'])
def reset_database():
    """Reset the database by clearing all queries and responses."""
    db = get_db()
    db.execute('DELETE FROM responses')
    db.execute('DELETE FROM queries')
    db.commit()
    db.close()
    
    return jsonify({
        'success': True,
        'message': 'Database has been reset. All queries and responses have been cleared.'
    })

@app.route('/health')
def health_check():
    """Health check endpoint for Render"""
    return jsonify({
        'status': 'healthy',
        'ai_systems_configured': {
            'openai': OPENAI_API_KEY is not None,
            'google': GOOGLE_API_KEY is not None,
            'anthropic': ANTHROPIC_API_KEY is not None,
            'mistral': MISTRAL_API_KEY is not None,
            'deepseek': DEEPSEEK_API_KEY is not None,
            'cohere': COHERE_API_KEY is not None,
            'groq': GROQ_API_KEY is not None,
            'ai21': AI21_API_KEY is not None,
            'xai': XAI_API_KEY is not None
        },
        'total_systems': sum([
            OPENAI_API_KEY is not None,
            GOOGLE_API_KEY is not None,
            ANTHROPIC_API_KEY is not None,
            MISTRAL_API_KEY is not None,
            DEEPSEEK_API_KEY is not None,
            COHERE_API_KEY is not None,
            GROQ_API_KEY is not None,
            AI21_API_KEY is not None,
            XAI_API_KEY is not None
        ])
    })

@app.route('/export/csv')
def export_csv():
    """Export all queries and responses to CSV format for Google Sheets import.
    
    Downloads a CSV file with columns:
    Query ID, Question, Timestamp, AI System, Model, Rating, Response Time, Raw Response
    """
    import csv
    from io import StringIO
    from flask import make_response
    
    db = get_db()
    
    # Get all queries with their responses
    data = db.execute('''
        SELECT 
            q.id as query_id,
            q.question,
            q.timestamp as query_timestamp,
            r.ai_system,
            r.model,
            r.extracted_rating,
            r.response_time,
            r.raw_response,
            r.timestamp as response_timestamp
        FROM queries q
        LEFT JOIN responses r ON q.id = r.query_id
        ORDER BY q.timestamp DESC, r.id
    ''').fetchall()
    
    db.close()
    
    # Create CSV in memory
    output = StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow([
        'Query ID',
        'Question',
        'Query Date',
        'AI System',
        'Model',
        'Rating (0-10)',
        'Response Time (sec)',
        'Full Response',
        'Response Date'
    ])
    
    # Write data rows
    for row in data:
        writer.writerow([
            row['query_id'],
            row['question'],
            row['query_timestamp'],
            row['ai_system'] or '',
            row['model'] or '',
            row['extracted_rating'] if row['extracted_rating'] else '',
            round(row['response_time'], 2) if row['response_time'] else '',
            row['raw_response'] or '',
            row['response_timestamp'] or ''
        ])
    
    # Prepare response
    output.seek(0)
    response = make_response(output.getvalue())
    response.headers['Content-Type'] = 'text/csv'
    response.headers['Content-Disposition'] = f'attachment; filename=ai_bias_research_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    
    return response

@app.route('/export/summary-csv')
def export_summary_csv():
    """Export summary statistics per query to CSV.
    
    Downloads a CSV with one row per query showing:
    Query ID, Question, Date, # Responses, Avg Rating, Min Rating, Max Rating, Spread, Std Dev
    """
    import csv
    from io import StringIO
    from flask import make_response
    import math
    
    db = get_db()
    
    # Get all queries
    queries = db.execute('SELECT * FROM queries ORDER BY timestamp DESC').fetchall()
    
    output = StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow([
        'Query ID',
        'Question',
        'Date',
        'Responses Count',
        'Ratings Found',
        'Average Rating',
        'Min Rating',
        'Max Rating',
        'Spread',
        'Std Deviation',
        'AI Systems'
    ])
    
    # Write summary for each query
    for query in queries:
        # Get all responses for this query
        responses = db.execute('''
            SELECT ai_system, model, extracted_rating 
            FROM responses 
            WHERE query_id = ?
        ''', (query['id'],)).fetchall()
        
        ratings = [r['extracted_rating'] for r in responses if r['extracted_rating'] is not None]
        ai_systems = ', '.join([f"{r['ai_system']}-{r['model']}" for r in responses if r['ai_system']])
        
        if ratings:
            avg_rating = sum(ratings) / len(ratings)
            min_rating = min(ratings)
            max_rating = max(ratings)
            spread = max_rating - min_rating
            
            # Calculate standard deviation
            if len(ratings) > 1:
                variance = sum((r - avg_rating) ** 2 for r in ratings) / len(ratings)
                std_dev = math.sqrt(variance)
            else:
                std_dev = 0
            
            writer.writerow([
                query['id'],
                query['question'],
                query['timestamp'],
                len(responses),
                len(ratings),
                round(avg_rating, 3),
                round(min_rating, 3),
                round(max_rating, 3),
                round(spread, 3),
                round(std_dev, 3),
                ai_systems
            ])
        else:
            writer.writerow([
                query['id'],
                query['question'],
                query['timestamp'],
                len(responses),
                0,
                '',
                '',
                '',
                '',
                '',
                ai_systems
            ])
    
    db.close()
    
    # Prepare response
    output.seek(0)
    response = make_response(output.getvalue())
    response.headers['Content-Type'] = 'text/csv'
    response.headers['Content-Disposition'] = f'attachment; filename=ai_bias_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    
    return response

@app.route('/debug/test-ai21')
def debug_test_ai21():
    """Debug endpoint to test AI21 Jamba API configuration."""
    if not AI21_API_KEY:
        return jsonify({
            'status': 'error',
            'api_key_configured': False,
            'error_message': 'AI21_API_KEY environment variable not set',
            'suggestions': [
                'Add AI21_API_KEY to Render environment variables',
                'Get your API key from: https://studio.ai21.com/account/api-key'
            ]
        })
    
    if not ai21_client:
        return jsonify({
            'status': 'error',
            'api_key_configured': True,
            'error_message': 'AI21 client failed to initialize'
        })
    
    try:
        start_time = time.time()
        
        messages = [
            AI21ChatMessage(role="user", content="Say 'Hello' and nothing else.")
        ]
        
        response = ai21_client.chat.completions.create(
            model="jamba-1.5-large",
            messages=messages,
            max_tokens=50
        )
        
        response_time = time.time() - start_time
        
        return jsonify({
            'status': 'success',
            'api_key_configured': True,
            'api_key_prefix': AI21_API_KEY[:15] + '...',
            'model_tested': 'jamba-1.5-large',
            'response_time': round(response_time, 2),
            'response_preview': response.choices[0].message.content[:100]
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'api_key_configured': True,
            'error_message': str(e)
        })

@app.route('/debug/test-xai')
def debug_test_xai():
    """Debug endpoint to test xAI Grok API configuration."""
    if not XAI_API_KEY:
        return jsonify({
            'status': 'error',
            'api_key_configured': False,
            'error_message': 'XAI_API_KEY environment variable not set',
            'suggestions': [
                'Add XAI_API_KEY to Render environment variables',
                'Get your API key from: https://console.x.ai'
            ]
        })
    
    if not xai_client:
        return jsonify({
            'status': 'error',
            'api_key_configured': True,
            'error_message': 'xAI client failed to initialize'
        })
    
    try:
        start_time = time.time()
        response = xai_client.chat.completions.create(
            model="grok-beta",
            messages=[{"role": "user", "content": "Say 'Hello' and nothing else."}],
            max_tokens=50
        )
        response_time = time.time() - start_time
        
        return jsonify({
            'status': 'success',
            'api_key_configured': True,
            'api_key_prefix': XAI_API_KEY[:15] + '...',
            'model_tested': 'grok-beta',
            'response_time': round(response_time, 2),
            'response_preview': response.choices[0].message.content[:100]
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'api_key_configured': True,
            'error_message': str(e)
        })

@app.route('/debug/test-groq')
def debug_test_groq():
    """Debug endpoint to test Groq/Llama API configuration with UPDATED model."""
    if not GROQ_API_KEY:
        return jsonify({
            'status': 'error',
            'api_key_configured': False,
            'error_message': 'GROQ_API_KEY environment variable not set'
        })
    
    if not groq_client:
        return jsonify({
            'status': 'error',
            'api_key_configured': True,
            'error_message': 'Groq client failed to initialize'
        })
    
    try:
        start_time = time.time()
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "Say 'Hello' and nothing else."}],
            max_tokens=50
        )
        response_time = time.time() - start_time
        
        return jsonify({
            'status': 'success',
            'api_key_configured': True,
            'api_key_prefix': GROQ_API_KEY[:15] + '...',
            'model_tested': 'llama-3.3-70b-versatile',
            'model_note': 'UPDATED from deprecated llama-3.1-70b-versatile',
            'response_time': round(response_time, 2),
            'response_preview': response.choices[0].message.content[:100]
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'api_key_configured': True,
            'error_message': str(e)
        })

@app.route('/debug/test-deepseek')
def debug_test_deepseek():
    """Debug endpoint to test DeepSeek API configuration."""
    if not DEEPSEEK_API_KEY:
        return jsonify({
            'status': 'error',
            'api_key_configured': False,
            'error_message': 'DEEPSEEK_API_KEY environment variable not set'
        })
    
    if not deepseek_client:
        return jsonify({
            'status': 'error',
            'api_key_configured': True,
            'error_message': 'DeepSeek client failed to initialize'
        })
    
    try:
        start_time = time.time()
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "Say 'Hello' and nothing else."}],
            max_tokens=50
        )
        response_time = time.time() - start_time
        
        return jsonify({
            'status': 'success',
            'api_key_configured': True,
            'api_key_prefix': DEEPSEEK_API_KEY[:15] + '...',
            'model_tested': 'deepseek-chat',
            'response_time': round(response_time, 2),
            'response_preview': response.choices[0].message.content[:100]
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'api_key_configured': True,
            'error_message': str(e)
        })

@app.route('/debug/test-cohere')
def debug_test_cohere():
    """Debug endpoint to test Cohere API configuration with UPDATED model."""
    if not COHERE_API_KEY:
        return jsonify({
            'status': 'error',
            'api_key_configured': False,
            'error_message': 'COHERE_API_KEY environment variable not set'
        })
    
    try:
        start_time = time.time()
        
        url = "https://api.cohere.com/v2/chat"
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {COHERE_API_KEY}'
        }
        payload = {
            'model': 'command-a-03-2025',
            'messages': [{'role': 'user', 'content': "Say 'Hello' and nothing else."}],
            'max_tokens': 50
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            response_text = ''
            if 'message' in data and 'content' in data['message']:
                content = data['message']['content']
                if isinstance(content, list) and len(content) > 0:
                    response_text = content[0].get('text', '')[:100]
            
            return jsonify({
                'status': 'success',
                'api_key_configured': True,
                'api_key_prefix': COHERE_API_KEY[:15] + '...',
                'model_tested': 'command-a-03-2025',
                'model_note': 'UPDATED from deprecated command-r-plus',
                'response_time': round(response_time, 2),
                'response_preview': response_text
            })
        else:
            return jsonify({
                'status': 'error',
                'api_key_configured': True,
                'http_status': response.status_code,
                'error_message': response.text[:200]
            })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'api_key_configured': True,
            'error_message': str(e)
        })

@app.route('/debug/test-anthropic')
def debug_test_anthropic():
    """Debug endpoint to test Anthropic Claude API configuration."""
    if not ANTHROPIC_API_KEY:
        return jsonify({
            'status': 'error',
            'api_key_configured': False,
            'error_message': 'ANTHROPIC_API_KEY environment variable not set'
        })
    
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        'Content-Type': 'application/json',
        'x-api-key': ANTHROPIC_API_KEY,
        'anthropic-version': '2023-06-01'
    }
    payload = {
        'model': 'claude-sonnet-4-20250514',
        'max_tokens': 50,
        'messages': [{'role': 'user', 'content': 'Say "Hello" and nothing else.'}]
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            return jsonify({
                'status': 'success',
                'api_key_configured': True,
                'api_key_prefix': ANTHROPIC_API_KEY[:20] + '...',
                'model_tested': 'claude-sonnet-4-20250514',
                'http_status': 200,
                'response_preview': data.get('content', [{}])[0].get('text', '')[:100]
            })
        else:
            return jsonify({
                'status': 'error',
                'api_key_configured': True,
                'http_status': response.status_code,
                'error_message': response.text[:200]
            })
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'api_key_configured': True,
            'error_message': str(e)
        })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

# I did no harm and this file is not truncated
