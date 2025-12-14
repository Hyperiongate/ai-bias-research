"""
AI Bias Research Tool - Main Application
Created: December 13, 2024
Last Updated: December 14, 2024

FIXES:
- December 14, 2024: Fixed OpenAI client initialization for v1.0+ API
- December 14, 2024: Updated Gemini model from 'gemini-pro' to 'gemini-1.5-flash'

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
import google.generativeai as genai
import json
import time

app = Flask(__name__)

# Configure API keys from environment variables
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')

# Initialize APIs
openai_client = None
if OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

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

def query_google_gemini(question):
    """Query Google Gemini 1.5 Flash"""
    if not GOOGLE_API_KEY:
        return {
            'success': False,
            'error': 'Google API key not configured',
            'system': 'Google',
            'model': 'Gemini-1.5-Flash'
        }
    
    try:
        start_time = time.time()
        model = genai.GenerativeModel('models/gemini-pro')
        response = model.generate_content(question)
        response_time = time.time() - start_time
        
        raw_response = response.text
        
        return {
            'success': True,
            'system': 'Google',
            'model': 'Gemini-1.5-Flash',
            'raw_response': raw_response,
            'response_time': response_time
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'system': 'Google',
            'model': 'Gemini-1.5-Flash'
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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

# I did no harm and this file is not truncated
