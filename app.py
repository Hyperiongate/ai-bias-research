"""
AI Bias Research Tool - Main Application
Created: December 13, 2024
Last Updated: December 15, 2024

FIXES:
- December 14, 2024: Fixed OpenAI client initialization for v1.0+ API
- December 14, 2024: Updated Gemini model naming attempts
- December 14, 2024: Switched to direct REST API calls for Gemini
- December 15, 2024: Added model discovery - lists available models first, then uses correct one
- December 15, 2024: Added debug endpoint to check available Gemini models

This application queries multiple AI systems with the same question to detect bias patterns.
Designed for research purposes to cross-validate AI responses.

Author: Jim (Hyperiongate)
Purpose: Discover if there's "any there there" in AI bias detection
"""

from flask import Flask, render_template, request, jsonify
import os
import sqlite3
from datetime import datetime
from openai import OpenAI
import requests
import json
import time

app = Flask(__name__)

# Configure API keys from environment variables
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')

# Initialize OpenAI client
openai_client = None
if OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Database setup
DATABASE = 'bias_research.db'

def get_db():
    """Get database connection"""
    db = sqlite3.connect(DATABASE)
    db.row_factory = sqlite3.Row
    return db

def init_db():
    """Initialize database with schema"""
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
            extracted_rating INTEGER,
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
    """Query OpenAI GPT-4"""
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
    """Query OpenAI GPT-3.5 Turbo"""
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

def get_available_gemini_models():
    """Get list of available Gemini models that support generateContent."""
    if not GOOGLE_API_KEY:
        return []
    
    available_models = []
    
    # Try both API versions
    for api_version in ['v1beta', 'v1']:
        url = f"https://generativelanguage.googleapis.com/{api_version}/models"
        
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
                    
                    # Only include models that support generateContent
                    if 'generateContent' in supported_methods:
                        # Extract just the model ID (e.g., "gemini-pro" from "models/gemini-pro")
                        model_id = model_name.replace('models/', '')
                        available_models.append({
                            'api_version': api_version,
                            'model_id': model_id,
                            'display_name': model.get('displayName', model_id),
                            'full_name': model_name
                        })
                
                # If we found models, no need to check other API version
                if available_models:
                    break
                    
        except Exception as e:
            continue
    
    return available_models

def query_google_gemini(question):
    """Query Google Gemini using direct REST API calls.
    
    First discovers available models, then uses the best one.
    Prefers gemini-1.5-flash or gemini-pro.
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
        
        # Get available models
        available_models = get_available_gemini_models()
        
        if not available_models:
            return {
                'success': False,
                'error': 'No Gemini models available. Check API key permissions.',
                'system': 'Google',
                'model': 'Gemini'
            }
        
        # Prefer these models in order
        preferred_models = ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro', 'gemini-1.0-pro']
        
        # Find the best available model
        selected_model = None
        for preferred in preferred_models:
            for model in available_models:
                if preferred in model['model_id']:
                    selected_model = model
                    break
            if selected_model:
                break
        
        # If no preferred model found, use first available
        if not selected_model and available_models:
            selected_model = available_models[0]
        
        if not selected_model:
            return {
                'success': False,
                'error': 'No suitable Gemini model found',
                'system': 'Google',
                'model': 'Gemini'
            }
        
        # Make the API call
        api_version = selected_model['api_version']
        model_id = selected_model['model_id']
        
        url = f"https://generativelanguage.googleapis.com/{api_version}/models/{model_id}:generateContent"
        
        headers = {
            'Content-Type': 'application/json'
        }
        
        payload = {
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
                        'model': selected_model.get('display_name', model_id),
                        'raw_response': raw_response,
                        'response_time': response_time
                    }
            
            return {
                'success': False,
                'error': f"Unexpected response format from {model_id}",
                'system': 'Google',
                'model': model_id
            }
        else:
            try:
                error_data = response.json()
                error_msg = error_data.get('error', {}).get('message', f"HTTP {response.status_code}")
            except:
                error_msg = f"HTTP {response.status_code}: {response.text[:200]}"
            
            return {
                'success': False,
                'error': f"{model_id}: {error_msg}",
                'system': 'Google',
                'model': model_id
            }
        
    except requests.exceptions.Timeout:
        return {
            'success': False,
            'error': 'Request timed out after 30 seconds',
            'system': 'Google',
            'model': 'Gemini'
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'system': 'Google',
            'model': 'Gemini'
        }

def extract_rating(text):
    """
    Attempt to extract a numerical rating from the response.
    Looks for patterns like "7/10", "7 out of 10", "rating: 7", etc.
    Returns None if no clear rating found.
    """
    import re
    
    # Pattern 1: X/10 or X out of 10
    pattern1 = r'\b(\d+)\s*(?:/|out of)\s*10\b'
    match = re.search(pattern1, text, re.IGNORECASE)
    if match:
        rating = int(match.group(1))
        if 0 <= rating <= 10:
            return rating
    
    # Pattern 2: "rating: X" or "score: X"
    pattern2 = r'(?:rating|score)[\s:]+(\d+)'
    match = re.search(pattern2, text, re.IGNORECASE)
    if match:
        rating = int(match.group(1))
        if 0 <= rating <= 10:
            return rating
    
    # Pattern 3: Single digit at start of response (risky, but common)
    pattern3 = r'^(\d+)\b'
    match = re.search(pattern3, text.strip(), re.IGNORECASE)
    if match:
        rating = int(match.group(1))
        if 0 <= rating <= 10:
            return rating
    
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

@app.route('/health')
def health_check():
    """Health check endpoint for Render"""
    return jsonify({
        'status': 'healthy',
        'openai_configured': OPENAI_API_KEY is not None,
        'google_configured': GOOGLE_API_KEY is not None
    })

@app.route('/debug/gemini-models')
def debug_gemini_models():
    """Debug endpoint to check what Gemini models are available.
    
    Visit /debug/gemini-models to see what models your API key can access.
    """
    if not GOOGLE_API_KEY:
        return jsonify({
            'error': 'Google API key not configured',
            'available_models': []
        })
    
    models = get_available_gemini_models()
    
    # Also try to get raw response from list models endpoint
    raw_responses = {}
    for api_version in ['v1beta', 'v1']:
        url = f"https://generativelanguage.googleapis.com/{api_version}/models"
        try:
            response = requests.get(
                url,
                params={'key': GOOGLE_API_KEY},
                timeout=10
            )
            raw_responses[api_version] = {
                'status_code': response.status_code,
                'response': response.json() if response.status_code == 200 else response.text[:500]
            }
        except Exception as e:
            raw_responses[api_version] = {'error': str(e)}
    
    return jsonify({
        'google_api_key_configured': True,
        'google_api_key_prefix': GOOGLE_API_KEY[:10] + '...' if GOOGLE_API_KEY else None,
        'available_models_for_generateContent': models,
        'raw_api_responses': raw_responses
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

# I did no harm and this file is not truncated
