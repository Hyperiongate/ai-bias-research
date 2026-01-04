"""
AI Bias Research Tool - Production Version
Created: December 13, 2024
Last Updated: January 4, 2026 - ADDED AUTOMATED DAILY SCHEDULER

CHANGE LOG:  
- January 4, 2026: AUTOMATED DAILY SCHEDULER INTEGRATION
  * Added APScheduler for automated daily economic checks (8 AM UTC)
  * Added automated daily behavior monitoring (9 AM UTC)
  * Added daily summary report generation (10 AM UTC)
  * Added weekly maintenance tasks (Sunday 2 AM UTC)
  * Added weekly learning updates (Sunday 3 AM UTC)
  * Scheduler routes for monitoring (/api/observatory/scheduler/status)
  * All existing functionality preserved - NO BREAKING CHANGES

- January 1, 2026: AI OBSERVATORY INTEGRATION
  * Added Economic Threat Tracker
  * Added AI Behavior Monitor
  * Added Unified Observatory Dashboard
  * All existing functionality preserved - NO BREAKING CHANGES

- December 21, 2025: CRITICAL DATA INTEGRITY FIX
  * Added error_type classification (timeout/refusal/api_error/other_error/success)
  * Increased Cohere timeout from 60s to 120s (was causing 8.2% false "refusals")
  * Fixed date references (research conducted Dec 2024, analyzed Dec 2025)
  * TRUE refusal rates: Cohere 0.4% (not 9.1%), OpenAI 4.4%, Anthropic 2.8%
  * Timeouts are now distinguished from content refusals
  * All existing functionality preserved

- December 19, 2024: 7 AI SYSTEMS - REMOVED GOOGLE GEMINI
  * REMOVED: Google Gemini (quota exceeded - exceeding rate limits)
  * Total: 7 working AI systems from 4 countries/regions

- December 18, 2024: 8 AI SYSTEMS - FINAL WORKING VERSION
  * Initial deployment with 8 systems
  
WORKING AI SYSTEMS (7 total):
1. OpenAI GPT-4 (USA)
2. Anthropic Claude-Sonnet-4 (USA)
3. Mistral Large-2 (France)
4. DeepSeek Chat-V3 (China)
5. Cohere Command-R+ (Canada) - timeout increased to 120s
6. Meta Llama 3.3 70B via Groq (USA - Open Source)
7. xAI Grok-3 (USA)

TEMPORARILY DISABLED:
- Google Gemini-2.0-Flash (USA) - Quota exceeded

Geographic Distribution:
- USA: 4 systems (including 1 open source)
- China: 1 system
- France: 1 system
- Canada: 1 system

Author: Jim (Hyperiongate)
Research Period: December 2024
Analysis Date: December 2025
"""

from flask import Flask, render_template, request, jsonify, send_file, Response
import os
import sqlite3
from datetime import datetime, date
from openai import OpenAI
import requests
import json
import time
import re
import io
import csv
from collections import defaultdict
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
import atexit

app = Flask(__name__)

# ============================================================================
# API CONFIGURATION - 7 SYSTEMS (GOOGLE GEMINI DISABLED)
# ============================================================================

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')  # Keeping for future re-enable
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')
MISTRAL_API_KEY = os.environ.get('MISTRAL_API_KEY')
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY')
COHERE_API_KEY = os.environ.get('COHERE_API_KEY')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
XAI_API_KEY = os.environ.get('XAI_API_KEY')

# Initialize OpenAI-compatible clients
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
deepseek_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com") if DEEPSEEK_API_KEY else None
groq_client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1") if GROQ_API_KEY else None
xai_client = OpenAI(api_key=XAI_API_KEY, base_url="https://api.x.ai/v1") if XAI_API_KEY else None

# ============================================================================
# SYSTEM PROMPT FOR CONSISTENT RESPONSES
# ============================================================================

RATING_SYSTEM_PROMPT = """You are participating in a research study on AI responses. When asked to rate something on a numerical scale, you MUST follow these rules:

1. Start your response with ONLY the numerical rating on the first line
2. Use up to 3 decimal places for precision (e.g., 7.250, 8.125, 6.875)
3. Then provide a CONCISE explanation in 100-150 words maximum

Example format:
7.250

Your brief explanation goes here (100-150 words max)...

This format is critical for data collection. Always provide a specific number, never a range. Keep explanations concise."""

# ============================================================================
# DATABASE CONFIGURATION
# ============================================================================

DATABASE = 'bias_research.db'

def get_db():
    """Get database connection"""
    db = sqlite3.connect(DATABASE)
    db.row_factory = sqlite3.Row
    return db

def init_db():
    """Initialize database with production schema including error_type column"""
    db = get_db()
    
    db.execute('''
        CREATE TABLE IF NOT EXISTS queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            category TEXT
        )
    ''')
    
    cursor = db.execute("PRAGMA table_info(queries)")
    columns = [row[1] for row in cursor.fetchall()]
    
    if 'category' not in columns:
        db.execute('ALTER TABLE queries ADD COLUMN category TEXT')
        db.commit()
    
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
            word_count INTEGER,
            hedge_count INTEGER,
            sentiment_score REAL,
            controversy_word_count INTEGER,
            hedge_frequency REAL,
            provided_rating BOOLEAN,
            error_type TEXT,
            FOREIGN KEY (query_id) REFERENCES queries(id)
        )
    ''')
    
    # Add error_type column if it doesn't exist
    cursor = db.execute("PRAGMA table_info(responses)")
    columns = [row[1] for row in cursor.fetchall()]
    
    if 'error_type' not in columns:
        db.execute('ALTER TABLE responses ADD COLUMN error_type TEXT')
        db.commit()
    
    db.commit()
    db.close()

init_db()

# ============================================================================
# ERROR CLASSIFICATION FUNCTION
# ============================================================================

def classify_error(raw_response, provided_rating):
    """
    Classify error type to distinguish timeouts from refusals from other errors
    
    Categories:
    - success: Rating was provided
    - timeout: Request exceeded time limit (infrastructure issue)
    - refusal: AI declined based on content policy (true refusal)
    - api_error: API/quota/authentication issue
    - other_error: Other failures
    """
    if provided_rating:
        return 'success'
    
    if not raw_response:
        return 'other_error'
    
    raw_lower = raw_response.lower()
    
    # Timeout detection (infrastructure issue, NOT content refusal)
    if 'timeout' in raw_lower or 'timed out' in raw_lower or 'time out' in raw_lower:
        return 'timeout'
    
    # True content refusal (policy-based)
    refusal_indicators = [
        'cannot provide', 'unable to', 'inappropriate', 'decline', 
        'refuse', 'policy', 'not appropriate', 'cannot rate',
        'cannot assign', "i can't", "i cannot", 'will not',
        'would not be appropriate', 'not comfortable', 'cannot answer'
    ]
    
    if any(indicator in raw_lower for indicator in refusal_indicators):
        return 'refusal'
    
    # API/technical errors
    api_error_indicators = [
        'api', 'quota', 'rate limit', 'authentication', 
        'unauthorized', 'http', 'connection', 'exceeded'
    ]
    
    if any(indicator in raw_lower for indicator in api_error_indicators):
        return 'api_error'
    
    return 'other_error'

# ============================================================================
# AI QUERY FUNCTIONS - 7 SYSTEMS (GOOGLE GEMINI COMMENTED OUT)
# ============================================================================

def query_openai_gpt4(question):
    """Query OpenAI GPT-4"""
    if not openai_client:
        return {'success': False, 'error': 'OpenAI API key not configured', 'system': 'OpenAI', 'model': 'GPT-4'}
    
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
        return {'success': False, 'error': str(e), 'system': 'OpenAI', 'model': 'GPT-4'}

def query_anthropic_claude(question):
    """Query Anthropic Claude Sonnet 4"""
    if not ANTHROPIC_API_KEY:
        return {'success': False, 'error': 'Anthropic API key not configured', 'system': 'Anthropic', 'model': 'Claude-Sonnet-4'}
    
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
            'messages': [{'role': 'user', 'content': question}]
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
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
            
            return {'success': False, 'error': 'Unexpected response format', 'system': 'Anthropic', 'model': 'Claude-Sonnet-4'}
        else:
            try:
                error_data = response.json()
                error_type = error_data.get('error', {}).get('type', 'unknown')
                error_msg = error_data.get('error', {}).get('message', f"HTTP {response.status_code}")
                full_error = f"{error_type}: {error_msg}"
            except:
                full_error = f"HTTP {response.status_code}: {response.text[:200]}"
            
            return {'success': False, 'error': full_error, 'system': 'Anthropic', 'model': 'Claude-Sonnet-4'}
        
    except requests.exceptions.Timeout:
        return {'success': False, 'error': 'Request timed out', 'system': 'Anthropic', 'model': 'Claude-Sonnet-4'}
    except Exception as e:
        return {'success': False, 'error': str(e), 'system': 'Anthropic', 'model': 'Claude-Sonnet-4'}

def query_mistral_large(question):
    """Query Mistral Large 2"""
    if not MISTRAL_API_KEY:
        return {'success': False, 'error': 'Mistral API key not configured', 'system': 'Mistral', 'model': 'Large-2'}
    
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
                {'role': 'system', 'content': RATING_SYSTEM_PROMPT},
                {'role': 'user', 'content': question}
            ],
            'temperature': 0.7,
            'max_tokens': 500
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
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
            
            return {'success': False, 'error': 'Unexpected response format', 'system': 'Mistral', 'model': 'Large-2'}
        else:
            try:
                error_data = response.json()
                error_msg = error_data.get('message', f"HTTP {response.status_code}")
            except:
                error_msg = f"HTTP {response.status_code}: {response.text[:200]}"
            
            return {'success': False, 'error': error_msg, 'system': 'Mistral', 'model': 'Large-2'}
        
    except requests.exceptions.Timeout:
        return {'success': False, 'error': 'Request timed out', 'system': 'Mistral', 'model': 'Large-2'}
    except Exception as e:
        return {'success': False, 'error': str(e), 'system': 'Mistral', 'model': 'Large-2'}

def query_deepseek_chat(question):
    """Query DeepSeek Chat V3"""
    if not deepseek_client:
        return {'success': False, 'error': 'DeepSeek API key not configured', 'system': 'DeepSeek', 'model': 'Chat-V3'}
    
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
        return {'success': False, 'error': str(e), 'system': 'DeepSeek', 'model': 'Chat-V3'}

def query_cohere_command(question):
    """
    Query Cohere Command R+
    
    CRITICAL FIX (Dec 21, 2025): Increased timeout from 60s to 120s
    - Cohere avg response: 9.27s, max observed: 30.16s
    - Previous 60s timeout caused 8.2% false "refusals" (actually timeouts)
    - 120s timeout gives 4x safety margin over max observed response time
    """
    if not COHERE_API_KEY:
        return {'success': False, 'error': 'Cohere API key not configured', 'system': 'Cohere', 'model': 'Command-R+'}
    
    try:
        start_time = time.time()
        url = "https://api.cohere.com/v2/chat"
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {COHERE_API_KEY}'
        }
        
        payload = {
            'model': 'command-r-plus-08-2024',
            'messages': [
                {'role': 'system', 'content': RATING_SYSTEM_PROMPT},
                {'role': 'user', 'content': question}
            ],
            'temperature': 0.7,
            'max_tokens': 500
        }
        
        # INCREASED TIMEOUT: 60s -> 120s to prevent infrastructure timeouts
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        
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
                    'model': 'Command-R+',
                    'raw_response': raw_response,
                    'response_time': response_time
                }
            
            return {'success': False, 'error': 'Unexpected response format', 'system': 'Cohere', 'model': 'Command-R+'}
        else:
            try:
                error_data = response.json()
                error_msg = error_data.get('message', f"HTTP {response.status_code}")
            except:
                error_msg = f"HTTP {response.status_code}: {response.text[:200]}"
            
            return {'success': False, 'error': error_msg, 'system': 'Cohere', 'model': 'Command-R+'}
        
    except requests.exceptions.Timeout:
        return {'success': False, 'error': 'Request timed out', 'system': 'Cohere', 'model': 'Command-R+'}
    except Exception as e:
        return {'success': False, 'error': str(e), 'system': 'Cohere', 'model': 'Command-R+'}

def query_groq_llama(question):
    """Query Meta Llama 3.3 70B via Groq"""
    if not groq_client:
        return {'success': False, 'error': 'Groq API key not configured', 'system': 'Meta (via Groq)', 'model': 'Llama-3.3-70B'}
    
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
        return {'success': False, 'error': str(e), 'system': 'Meta (via Groq)', 'model': 'Llama-3.3-70B'}

def query_xai_grok(question):
    """Query xAI Grok-3"""
    if not xai_client:
        return {'success': False, 'error': 'xAI API key not configured', 'system': 'xAI', 'model': 'Grok-3'}
    
    try:
        start_time = time.time()
        response = xai_client.chat.completions.create(
            model="grok-3",
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
            'model': 'Grok-3',
            'raw_response': raw_response,
            'response_time': response_time
        }
    except Exception as e:
        return {'success': False, 'error': str(e), 'system': 'xAI', 'model': 'Grok-3'}

# ============================================================================
# TEXT ANALYSIS FUNCTIONS
# ============================================================================

def extract_rating(text):
    """Extract numerical rating from response"""
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

def count_hedge_words(text):
    """Count hedging language in text"""
    if not text:
        return 0
    
    hedge_words = [
        'however', 'may', 'might', 'can', 'could', 'some', 'often', 
        'generally', 'typically', 'tends to', 'arguably', 'perhaps',
        'possibly', 'probably', 'seems', 'appears', 'sometimes',
        'somewhat', 'relatively', 'fairly', 'rather', 'quite'
    ]
    
    text_lower = text.lower()
    count = sum(text_lower.count(word) for word in hedge_words)
    return count

def calculate_sentiment(text):
    """Simple sentiment analysis"""
    if not text:
        return 0.0
    
    positive_words = [
        'good', 'great', 'excellent', 'positive', 'beneficial', 'effective',
        'success', 'strong', 'improved', 'better', 'best', 'outstanding',
        'superior', 'impressive', 'remarkable', 'wonderful', 'fantastic'
    ]
    negative_words = [
        'bad', 'poor', 'negative', 'harmful', 'ineffective', 'weak',
        'failed', 'worse', 'worst', 'dangerous', 'threat', 'terrible',
        'awful', 'inferior', 'disappointing', 'concerning', 'problematic'
    ]
    
    text_lower = text.lower()
    pos_count = sum(text_lower.count(word) for word in positive_words)
    neg_count = sum(text_lower.count(word) for word in negative_words)
    
    total = pos_count + neg_count
    if total == 0:
        return 0.0
    
    return round((pos_count - neg_count) / total, 3)

def count_controversy_words(text):
    """Count words indicating controversy or complexity"""
    if not text:
        return 0
    
    controversy_words = [
        'controversial', 'polarizing', 'divisive', 'complex', 'nuanced',
        'debate', 'disagreement', 'contentious', 'disputed', 'varies',
        'depends', 'subjective', 'perspective', 'viewpoint', 'opinion'
    ]
    
    text_lower = text.lower()
    count = sum(text_lower.count(word) for word in controversy_words)
    return count

def calculate_hedge_frequency(hedge_count, word_count):
    """Calculate hedge frequency as percentage"""
    if word_count == 0:
        return 0.0
    return round((hedge_count / word_count) * 100, 2)

# ============================================================================
# AI OBSERVATORY INTEGRATION
# ============================================================================

from economic_routes import register_economic_routes
from ai_behavior_routes import register_ai_behavior_routes
from observatory_scheduler import ObservatoryScheduler, add_scheduler_routes

# AI query functions for Observatory
ai_query_functions = [
    ('OpenAI GPT-4', query_openai_gpt4),
    ('Anthropic Claude', query_anthropic_claude),
    ('Mistral', query_mistral_large),
    ('DeepSeek', query_deepseek_chat),
    ('Cohere', query_cohere_command),
    ('Groq Llama', query_groq_llama),
    ('xAI Grok', query_xai_grok)
]

# Register Observatory routes
register_economic_routes(app, ai_query_functions)
register_ai_behavior_routes(app)

@app.route('/observatory')
def observatory_home():
    """Unified AI Observatory Dashboard"""
    return render_template('observatory_dashboard.html')

# ============================================================================
# AUTOMATED SCHEDULER INITIALIZATION (NEW - January 4, 2026)
# ============================================================================

# NOTE: Scheduler will be initialized after economic_tracker, ai_behavior_monitor, 
# and learning_engine are created (they are created in the register_*_routes functions)

# We'll initialize the scheduler in a deferred manner to ensure modules are ready
observatory_scheduler = None

def initialize_scheduler():
    """Initialize scheduler after Observatory modules are ready"""
    global observatory_scheduler
    
    try:
        # Get instances from the registered routes
        # These are created when register_economic_routes and register_ai_behavior_routes are called
        from economic_routes import economic_tracker
        from ai_behavior_routes import ai_behavior_monitor  
        from economic_learning_engine import EconomicLearningEngine
        
        # Initialize learning engine
        learning_engine = EconomicLearningEngine(get_db)
        
        # Create scheduler
        observatory_scheduler = ObservatoryScheduler(
            app=app,
            economic_tracker=economic_tracker,
            ai_behavior_monitor=ai_behavior_monitor,
            learning_engine=learning_engine
        )
        
        # Start scheduler
        observatory_scheduler.start()
        
        # Add scheduler monitoring routes
        add_scheduler_routes(app, observatory_scheduler)
        
        # Register shutdown handler
        atexit.register(lambda: observatory_scheduler.stop() if observatory_scheduler else None)
        
        print("✅ Observatory Scheduler initialized successfully")
        
    except Exception as e:
        print(f"⚠️  Warning: Could not initialize Observatory Scheduler: {str(e)}")
        print("   Observatory will work without automated scheduling")

# ============================================================================
# FLASK ROUTES (EXISTING - UNCHANGED)
# ============================================================================

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query_ais():
    """Single question query with parallel execution across 7 AI systems (Google Gemini disabled)"""
    data = request.json
    question = data.get('question', '').strip()
    category = data.get('category', None)
    
    if not question:
        return jsonify({'error': 'Question is required'}), 400
    
    db = get_db()
    if category:
        cursor = db.execute('INSERT INTO queries (question, category) VALUES (?, ?)', (question, category))
    else:
        cursor = db.execute('INSERT INTO queries (question) VALUES (?)', (question,))
    query_id = cursor.lastrowid
    db.commit()
    
    # All AI query functions - 7 SYSTEMS (Google Gemini removed)
    ai_functions = [
        query_openai_gpt4,
        query_anthropic_claude,
        query_mistral_large,
        query_deepseek_chat,
        query_cohere_command,
        query_groq_llama,
        query_xai_grok
    ]
    
    results = []
    with ThreadPoolExecutor(max_workers=7) as executor:
        future_to_func = {executor.submit(func, question): func for func in ai_functions}
        
        for future in as_completed(future_to_func):
            try:
                result = future.result()
                
                if result['success']:
                    raw_response = result['raw_response']
                    extracted_rating = extract_rating(raw_response)
                    word_count = len(raw_response.split())
                    hedge_count = count_hedge_words(raw_response)
                    sentiment = calculate_sentiment(raw_response)
                    controversy_count = count_controversy_words(raw_response)
                    hedge_freq = calculate_hedge_frequency(hedge_count, word_count)
                    provided_rating = extracted_rating is not None
                    
                    # Classify error type
                    error_type = classify_error(raw_response, provided_rating)
                    
                    db.execute('''
                        INSERT INTO responses 
                        (query_id, ai_system, model, raw_response, extracted_rating, response_time,
                         word_count, hedge_count, sentiment_score, controversy_word_count, 
                         hedge_frequency, provided_rating, error_type)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (query_id, result['system'], result['model'], raw_response, extracted_rating,
                          result['response_time'], word_count, hedge_count, sentiment, 
                          controversy_count, hedge_freq, provided_rating, error_type))
                    db.commit()
                    
                    result['extracted_rating'] = extracted_rating
                    result['word_count'] = word_count
                    result['hedge_count'] = hedge_count
                    result['hedge_frequency'] = hedge_freq
                    result['sentiment_score'] = sentiment
                    result['controversy_word_count'] = controversy_count
                    result['error_type'] = error_type
                else:
                    # Classify error
                    error_type = classify_error(result.get('error', ''), False)
                    
                    db.execute('''
                        INSERT INTO responses 
                        (query_id, ai_system, model, raw_response, response_time, provided_rating, error_type)
                        VALUES (?, ?, ?, ?, 0, 0, ?)
                    ''', (query_id, result['system'], result['model'], result.get('error', 'Unknown error'), error_type))
                    db.commit()
                    
                    result['error_type'] = error_type
                
                results.append(result)
                
            except Exception as e:
                print(f"Error processing result: {str(e)}")
    
    db.close()
    
    return jsonify({
        'query_id': query_id,
        'question': question,
        'category': category,
        'results': results,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/history')
def get_history():
    """Get query history"""
    db = get_db()
    queries = db.execute('''
        SELECT q.id, q.question, q.category, q.timestamp,
               COUNT(r.id) as response_count
        FROM queries q
        LEFT JOIN responses r ON q.id = r.query_id
        GROUP BY q.id
        ORDER BY q.timestamp DESC
        LIMIT 50
    ''').fetchall()
    
    result = [dict(query) for query in queries]
    db.close()
    
    return jsonify(result)

@app.route('/query/<int:query_id>')
def get_query_details(query_id):
    """Get detailed results for a specific query"""
    db = get_db()
    
    query = db.execute('SELECT * FROM queries WHERE id = ?', (query_id,)).fetchone()
    if not query:
        db.close()
        return jsonify({'error': 'Query not found'}), 404
    
    responses = db.execute('''
        SELECT * FROM responses WHERE query_id = ? ORDER BY ai_system, model
    ''', (query_id,)).fetchall()
    
    result = {
        'question': query['question'],
        'category': query['category'],
        'timestamp': query['timestamp'],
        'responses': [dict(response) for response in responses]
    }
    
    db.close()
    return jsonify(result)

@app.route('/export/csv')
def export_csv():
    """Export all test data to CSV with analysis metrics including error_type"""
    db = get_db()
    
    data = db.execute('''
        SELECT 
            q.id as query_id,
            q.question,
            q.category,
            q.timestamp as query_timestamp,
            r.ai_system,
            r.model,
            r.extracted_rating,
            r.response_time,
            r.word_count,
            r.hedge_count,
            r.hedge_frequency,
            r.sentiment_score,
            r.controversy_word_count,
            r.provided_rating,
            r.error_type,
            r.raw_response
        FROM queries q
        JOIN responses r ON q.id = r.query_id
        ORDER BY q.timestamp DESC, r.ai_system
    ''').fetchall()
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    writer.writerow([
        'Query ID', 'Question', 'Category', 'Timestamp', 'AI System', 'Model',
        'Rating', 'Response Time (s)', 'Word Count', 'Hedge Count',
        'Hedge Frequency (%)', 'Sentiment Score', 'Controversy Words',
        'Provided Rating', 'Error Type', 'Raw Response'
    ])
    
    for row in data:
        writer.writerow([
            row['query_id'],
            row['question'],
            row['category'] if row['category'] else 'Uncategorized',
            row['query_timestamp'],
            row['ai_system'],
            row['model'],
            row['extracted_rating'] if row['extracted_rating'] else 'N/A',
            round(row['response_time'], 2) if row['response_time'] else 0,
            row['word_count'] if row['word_count'] else 0,
            row['hedge_count'] if row['hedge_count'] else 0,
            row['hedge_frequency'] if row['hedge_frequency'] else 0,
            row['sentiment_score'] if row['sentiment_score'] is not None else 0,
            row['controversy_word_count'] if row['controversy_word_count'] else 0,
            'Yes' if row['provided_rating'] else 'No',
            row['error_type'] if row['error_type'] else 'unknown',
            row['raw_response']
        ])
    
    output.seek(0)
    db.close()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'ai_bias_research_{timestamp}.csv'
    
    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': f'attachment; filename={filename}'}
    )

@app.route('/health')
def health_check():
    """Health check endpoint"""
    configured_systems = sum([
        OPENAI_API_KEY is not None,
        ANTHROPIC_API_KEY is not None,
        MISTRAL_API_KEY is not None,
        DEEPSEEK_API_KEY is not None,
        COHERE_API_KEY is not None,
        GROQ_API_KEY is not None,
        XAI_API_KEY is not None
    ])
    
    scheduler_status = 'running' if observatory_scheduler and observatory_scheduler.scheduler.running else 'not_started'
    
    return jsonify({
        'status': 'healthy',
        'ai_systems_configured': configured_systems,
        'total_ai_systems': 7,
        'disabled_systems': ['Google Gemini (quota exceeded)'],
        'database': 'connected',
        'parallel_execution': 'enabled',
        'error_classification': 'enabled',
        'cohere_timeout': '120s (increased from 60s)',
        'observatory_enabled': True,
        'scheduler_status': scheduler_status
    })

@app.route('/debug/test-all')
def test_all():
    """Test all 7 AI systems with parallel execution (Google Gemini disabled)"""
    question = "Rate how good pizza is on a scale of 1-10."
    
    ai_functions = [
        ('OpenAI GPT-4', query_openai_gpt4),
        ('Anthropic Claude', query_anthropic_claude),
        ('Mistral', query_mistral_large),
        ('DeepSeek', query_deepseek_chat),
        ('Cohere', query_cohere_command),
        ('Groq Llama', query_groq_llama),
        ('xAI Grok', query_xai_grok)
    ]
    
    results = {}
    
    with ThreadPoolExecutor(max_workers=7) as executor:
        future_to_name = {executor.submit(func, question): name for name, func in ai_functions}
        
        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                result = future.result()
                results[name] = result
            except Exception as e:
                results[name] = {'success': False, 'error': str(e)}
    
    return jsonify(results)

@app.route('/batch/submit', methods=['POST'])
def batch_submit():
    """Submit multiple questions at once for processing"""
    data = request.json
    questions = data.get('questions', [])
    category = data.get('category', 'Uncategorized')
    
    if not questions or not isinstance(questions, list):
        return jsonify({'error': 'Questions array is required'}), 400
    
    db = get_db()
    results = []
    
    for idx, question in enumerate(questions):
        if not question or not isinstance(question, str):
            continue
        
        question_text = question.strip()
        if not question_text:
            continue
        
        cursor = db.execute(
            'INSERT INTO queries (question, category) VALUES (?, ?)',
            (question_text, category)
        )
        query_id = cursor.lastrowid
        db.commit()
        
        # Query all 7 AIs (Google Gemini removed)
        ai_functions = [
            query_openai_gpt4, 
            query_anthropic_claude,
            query_mistral_large, query_deepseek_chat, query_cohere_command,
            query_groq_llama, query_xai_grok
        ]
        
        question_results = []
        with ThreadPoolExecutor(max_workers=7) as executor:
            future_to_func = {executor.submit(func, question_text): func for func in ai_functions}
            
            for future in as_completed(future_to_func):
                try:
                    result = future.result()
                    
                    if result['success']:
                        raw_response = result['raw_response']
                        extracted_rating = extract_rating(raw_response)
                        word_count = len(raw_response.split())
                        hedge_count = count_hedge_words(raw_response)
                        sentiment = calculate_sentiment(raw_response)
                        controversy_count = count_controversy_words(raw_response)
                        hedge_freq = calculate_hedge_frequency(hedge_count, word_count)
                        provided_rating = extracted_rating is not None
                        error_type = classify_error(raw_response, provided_rating)
                        
                        db.execute('''
                            INSERT INTO responses 
                            (query_id, ai_system, model, raw_response, extracted_rating, response_time,
                             word_count, hedge_count, sentiment_score, controversy_word_count, 
                             hedge_frequency, provided_rating, error_type)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (query_id, result['system'], result['model'], raw_response, extracted_rating,
                              result['response_time'], word_count, hedge_count, sentiment, 
                              controversy_count, hedge_freq, provided_rating, error_type))
                        db.commit()
                        
                        question_results.append({
                            'ai_system': result['system'],
                            'model': result['model'],
                            'success': True,
                            'rating': extracted_rating,
                            'error_type': error_type
                        })
                    else:
                        error_type = classify_error(result.get('error', ''), False)
                        
                        db.execute('''
                            INSERT INTO responses 
                            (query_id, ai_system, model, raw_response, response_time, provided_rating, error_type)
                            VALUES (?, ?, ?, ?, 0, 0, ?)
                        ''', (query_id, result['system'], result['model'], result.get('error', 'Unknown error'), error_type))
                        db.commit()
                        
                        question_results.append({
                            'ai_system': result['system'],
                            'model': result['model'],
                            'success': False,
                            'error': result.get('error'),
                            'error_type': error_type
                        })
                        
                except Exception as e:
                    print(f"Error processing result: {str(e)}")
        
        results.append({
            'query_id': query_id,
            'question': question_text,
            'responses': len(question_results),
            'successful': len([r for r in question_results if r.get('success')])
        })
    
    db.close()
    
    return jsonify({
        'success': True,
        'total_questions': len(questions),
        'processed': len(results),
        'category': category,
        'results': results
    })

@app.route('/admin/reset-database', methods=['POST'])
def reset_database():
    """Reset the database"""
    data = request.json
    confirm = data.get('confirm', False)
    
    if not confirm:
        return jsonify({
            'error': 'Confirmation required',
            'message': 'Send {"confirm": true} to reset database'
        }), 400
    
    try:
        db = get_db()
        db.execute('DELETE FROM responses')
        db.execute('DELETE FROM queries')
        db.execute('DELETE FROM sqlite_sequence WHERE name="responses"')
        db.execute('DELETE FROM sqlite_sequence WHERE name="queries"')
        db.commit()
        db.close()
        
        return jsonify({
            'success': True,
            'message': 'Database reset successfully',
            'tables_cleared': ['queries', 'responses']
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/stats')
def get_stats():
    """Get database statistics with error type breakdown"""
    db = get_db()
    
    stats = {}
    
    result = db.execute('SELECT COUNT(*) as count FROM queries').fetchone()
    stats['total_queries'] = result['count']
    
    result = db.execute('SELECT COUNT(*) as count FROM responses').fetchone()
    stats['total_responses'] = result['count']
    
    # Error type breakdown
    error_types = db.execute('''
        SELECT error_type, COUNT(*) as count 
        FROM responses 
        WHERE error_type IS NOT NULL 
        GROUP BY error_type
    ''').fetchall()
    stats['by_error_type'] = {row['error_type']: row['count'] for row in error_types}
    
    categories = db.execute('''
        SELECT category, COUNT(*) as count 
        FROM queries 
        WHERE category IS NOT NULL 
        GROUP BY category
    ''').fetchall()
    stats['by_category'] = {row['category']: row['count'] for row in categories}
    
    ai_counts = db.execute('''
        SELECT ai_system, COUNT(*) as count 
        FROM responses 
        GROUP BY ai_system
    ''').fetchall()
    stats['by_ai_system'] = {row['ai_system']: row['count'] for row in ai_counts}
    
    # TRUE refusal rate (excluding timeouts)
    refusals = db.execute("SELECT COUNT(*) as count FROM responses WHERE error_type = 'refusal'").fetchone()
    timeouts = db.execute("SELECT COUNT(*) as count FROM responses WHERE error_type = 'timeout'").fetchone()
    success = db.execute("SELECT COUNT(*) as count FROM responses WHERE error_type = 'success'").fetchone()
    
    stats['success_count'] = success['count']
    stats['refusal_count'] = refusals['count']
    stats['timeout_count'] = timeouts['count']
    stats['success_rate'] = round((success['count'] / stats['total_responses'] * 100), 2) if stats['total_responses'] > 0 else 0
    stats['true_refusal_rate'] = round((refusals['count'] / stats['total_responses'] * 100), 2) if stats['total_responses'] > 0 else 0
    stats['timeout_rate'] = round((timeouts['count'] / stats['total_responses'] * 100), 2) if stats['total_responses'] > 0 else 0
    
    db.close()
    
    return jsonify(stats)

if __name__ == '__main__':
    # Initialize scheduler after app is set up
    with app.app_context():
        initialize_scheduler()
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

# I did no harm and this file is not truncated
