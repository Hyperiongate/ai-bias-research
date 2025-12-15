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
                      and returning 404 not_found_error
- December 15, 2024: ADDED DEEPSEEK! Integrated DeepSeek AI from China (deepseek-chat model)
- December 15, 2024: ADDED COHERE! Integrated Cohere Command R+ from Canada
- December 15, 2024: ADDED LLAMA! Integrated Meta Llama 3.1 70B via Groq (open-source model)
                      Now have 8 AI systems: 4 US, 1 French, 1 Chinese, 1 Canadian, 1 Open-Source

This application queries multiple AI systems with the same question to detect bias patterns.
Designed for research purposes to cross-validate AI responses.

Author: Jim (Hyperiongate)
Purpose: Discover if there's "any there there" in AI bias detection

AI SYSTEMS INTEGRATED (8 total):
- OpenAI GPT-4 (USA) - Proprietary
- OpenAI GPT-3.5-Turbo (USA) - Proprietary
- Google Gemini-2.0-Flash (USA) - Proprietary
- Anthropic Claude-Sonnet-4 (USA) - Proprietary
- Mistral Large-2 (France) - Proprietary
- DeepSeek Chat (China) - Proprietary
- Cohere Command R+ (Canada) - Proprietary
- Meta Llama 3.1 70B via Groq (USA) - OPEN SOURCE - NEW!
"""

from flask import Flask, render_template, request, jsonify
import os
import sqlite3
from datetime import datetime
from openai import OpenAI
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
    Prepends system instructions to the question since Gemini handles
    system prompts differently than OpenAI.
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
        
        # Gemini uses systemInstruction field for system prompts
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
            
            # Extract text from response
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
                'error': f"Unexpected response format",
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
    
    Uses Claude Sonnet 4 via direct REST API calls (no SDK required).
    
    API Version: 2023-06-01
    Model: claude-sonnet-4-20250514 (Claude Sonnet 4 - released May 2025)
    
    Previous model claude-3-5-sonnet-20241022 was deprecated and returned 404.
    Updated December 15, 2024 to use Claude Sonnet 4.
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
        
        # Claude Sonnet 4 - current stable model as of December 2024
        # Previous model (claude-3-5-sonnet-20241022) was deprecated
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
            
            # Extract text from response
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
    Provides European perspective on AI responses.
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
            
            # Extract text from response
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
    
    DeepSeek is a Chinese AI company that released an open-source model
    matching GPT-4 performance at significantly lower cost.
    
    API Endpoint: https://api.deepseek.com
    Model: deepseek-chat (DeepSeek V3.2 non-thinking mode)
    
    Added December 15, 2024 for geographic diversity in bias research.
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
    """Query Cohere Command R+ with system prompt for structured responses.
    
    Uses Cohere Command R+ via REST API.
    Provides Canadian AI perspective on responses.
    
    Cohere is a Canadian AI company focused on enterprise NLP.
    Command R+ is their flagship model optimized for RAG and tool use.
    
    API Endpoint: https://api.cohere.com/v2/chat
    Model: command-r-plus
    
    Added December 15, 2024 for geographic diversity in bias research.
    """
    if not COHERE_API_KEY:
        return {
            'success': False,
            'error': 'Cohere API key not configured',
            'system': 'Cohere',
            'model': 'Command-R+'
        }
    
    try:
        start_time = time.time()
        
        url = "https://api.cohere.com/v2/chat"
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {COHERE_API_KEY}'
        }
        
        # Cohere v2 API uses a different format
        payload = {
            'model': 'command-r-plus',
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
            
            # Cohere v2 returns message.content array
            if 'message' in data and 'content' in data['message']:
                content = data['message']['content']
                if isinstance(content, list) and len(content) > 0:
                    raw_response = content[0].get('text', '')
                else:
                    raw_response = str(content)
                
                return {
                    'success': True,
                    'system': 'Cohere',
                    'model': 'Command-R+',
                    'raw_response': raw_response,
                    'response_time': response_time
                }
            
            return {
                'success': False,
                'error': 'Unexpected response format from Cohere API',
                'system': 'Cohere',
                'model': 'Command-R+'
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
                'model': 'Command-R+'
            }
        
    except requests.exceptions.Timeout:
        return {
            'success': False,
            'error': 'Request timed out after 30 seconds',
            'system': 'Cohere',
            'model': 'Command-R+'
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'system': 'Cohere',
            'model': 'Command-R+'
        }

def query_groq_llama(question):
    """Query Meta Llama 3.1 70B via Groq with system prompt for structured responses.
    
    Uses Llama 3.1 70B via Groq's ultra-fast LPU inference.
    This is an OPEN SOURCE model, unlike all other proprietary models in this tool.
    
    Groq provides the fastest inference for Llama models using their
    custom LPU (Language Processing Unit) hardware.
    
    API Endpoint: https://api.groq.com/openai/v1 (OpenAI-compatible)
    Model: llama-3.1-70b-versatile
    
    Added December 15, 2024 to include open-source model perspective.
    """
    if not groq_client:
        return {
            'success': False,
            'error': 'Groq API key not configured',
            'system': 'Meta',
            'model': 'Llama-3.1-70B'
        }
    
    try:
        start_time = time.time()
        response = groq_client.chat.completions.create(
            model="llama-3.1-70b-versatile",
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
            'model': 'Llama-3.1-70B',
            'raw_response': raw_response,
            'response_time': response_time
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'system': 'Meta (via Groq)',
            'model': 'Llama-3.1-70B'
        }

def extract_rating(text):
    """
    Extract numerical rating from the response.
    
    Since we're using system prompts that instruct the AI to put the number
    on the first line, this function is now much simpler - it just looks for
    the first number in the response.
    
    Returns float with up to 3 decimal places, or None if no rating found.
    """
    if not text:
        return None
    
    # Get the first line of the response
    first_line = text.strip().split('\n')[0].strip()
    
    # Look for a decimal number (e.g., 7.250, 8.5, 9)
    # Match numbers with optional decimal places
    match = re.search(r'^(\d+(?:\.\d+)?)', first_line)
    
    if match:
        try:
            rating = float(match.group(1))
            # Validate it's in the 0-10 range
            if 0 <= rating <= 10:
                # Round to 3 decimal places
                return round(rating, 3)
        except ValueError:
            pass
    
    # Fallback: search entire text for X/10 pattern
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
    """Query multiple AI systems with the same question"""
    data = request.json
    question = data.get('question', '').strip()
    
    if not question:
        return jsonify({'error': 'Question is required'}), 400
    
    # Save query to database
    db = get_db()
    cursor = db.execute('INSERT INTO queries (question) VALUES (?)', (question,))
    query_id = cursor.lastrowid
    db.commit()
    
    # Query all available AI systems
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
    
    # Cohere Command R+ (Canada)
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
    
    # Meta Llama 3.1 70B via Groq (Open Source)
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
    """Reset the database by clearing all queries and responses.
    
    This is useful during development/testing to start fresh.
    """
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
        'openai_configured': OPENAI_API_KEY is not None,
        'google_configured': GOOGLE_API_KEY is not None,
        'anthropic_configured': ANTHROPIC_API_KEY is not None,
        'mistral_configured': MISTRAL_API_KEY is not None,
        'deepseek_configured': DEEPSEEK_API_KEY is not None,
        'cohere_configured': COHERE_API_KEY is not None,
        'groq_configured': GROQ_API_KEY is not None
    })

@app.route('/debug/gemini-models')
def debug_gemini_models():
    """Debug endpoint to check what Gemini models are available."""
    if not GOOGLE_API_KEY:
        return jsonify({
            'error': 'Google API key not configured',
            'available_models': []
        })
    
    available_models = []
    
    url = "https://generativelanguage.googleapis.com/v1beta/models"
    try:
        response = requests.get(
            url,
            params={'key': GOOGLE_API_KEY},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            models = data.get('models', [])
            
            for model in models:
                model_name = model.get('name', '')
                supported_methods = model.get('supportedGenerationMethods', [])
                
                if 'generateContent' in supported_methods:
                    model_id = model_name.replace('models/', '')
                    available_models.append({
                        'model_id': model_id,
                        'display_name': model.get('displayName', model_id)
                    })
                    
    except Exception as e:
        return jsonify({'error': str(e)})
    
    return jsonify({
        'google_api_key_configured': True,
        'available_models': available_models,
        'recommended_model': 'gemini-2.0-flash'
    })

@app.route('/debug/test-anthropic')
def debug_test_anthropic():
    """Debug endpoint to test Anthropic Claude API configuration."""
    if not ANTHROPIC_API_KEY:
        return jsonify({
            'status': 'error',
            'api_key_configured': False,
            'error_message': 'ANTHROPIC_API_KEY environment variable not set',
            'suggestions': [
                'Add ANTHROPIC_API_KEY to Render environment variables',
                'Get your API key from: https://console.anthropic.com/settings/keys'
            ]
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
        'messages': [
            {
                'role': 'user',
                'content': 'Say "Hello, this is a test" and nothing else.'
            }
        ]
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
            try:
                error_data = response.json()
                error_type = error_data.get('error', {}).get('type', 'unknown')
                error_msg = error_data.get('error', {}).get('message', 'No message')
            except:
                error_type = 'unknown'
                error_msg = response.text[:200]
            
            return jsonify({
                'status': 'error',
                'api_key_configured': True,
                'http_status': response.status_code,
                'error_type': error_type,
                'error_message': error_msg
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
            'error_message': 'DEEPSEEK_API_KEY environment variable not set',
            'suggestions': [
                'Add DEEPSEEK_API_KEY to Render environment variables',
                'Get your API key from: https://platform.deepseek.com/api_keys'
            ]
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
    """Debug endpoint to test Cohere API configuration."""
    if not COHERE_API_KEY:
        return jsonify({
            'status': 'error',
            'api_key_configured': False,
            'error_message': 'COHERE_API_KEY environment variable not set',
            'suggestions': [
                'Add COHERE_API_KEY to Render environment variables',
                'Get your API key from: https://dashboard.cohere.com/api-keys'
            ]
        })
    
    try:
        start_time = time.time()
        
        url = "https://api.cohere.com/v2/chat"
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {COHERE_API_KEY}'
        }
        payload = {
            'model': 'command-r-plus',
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
                'model_tested': 'command-r-plus',
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

@app.route('/debug/test-groq')
def debug_test_groq():
    """Debug endpoint to test Groq/Llama API configuration."""
    if not GROQ_API_KEY:
        return jsonify({
            'status': 'error',
            'api_key_configured': False,
            'error_message': 'GROQ_API_KEY environment variable not set',
            'suggestions': [
                'Add GROQ_API_KEY to Render environment variables',
                'Get your API key from: https://console.groq.com/keys'
            ]
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
            model="llama-3.1-70b-versatile",
            messages=[{"role": "user", "content": "Say 'Hello' and nothing else."}],
            max_tokens=50
        )
        response_time = time.time() - start_time
        
        return jsonify({
            'status': 'success',
            'api_key_configured': True,
            'api_key_prefix': GROQ_API_KEY[:15] + '...',
            'model_tested': 'llama-3.1-70b-versatile',
            'response_time': round(response_time, 2),
            'response_preview': response.choices[0].message.content[:100]
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
