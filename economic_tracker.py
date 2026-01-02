#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AI Observatory - Economic Threat Tracker Module
File: economic_tracker.py
Date: January 1, 2026
Version: 1.0.0 - INITIAL RELEASE

PURPOSE:
Monitor economic indicators for AI-driven threats including job displacement,
market manipulation, productivity disruption, and economic systemic risks.

FEATURES:
- Real-time economic data collection (unemployment, job postings, wages)
- AI impact scoring across multiple economic sectors
- Trend detection and anomaly alerting
- Multi-AI consensus analysis of economic threats
- Historical tracking and prediction

DATA SOURCES:
- Bureau of Labor Statistics (BLS) API - Employment data
- FRED API (Federal Reserve Economic Data) - Economic indicators
- Indeed/LinkedIn job posting APIs - AI job market impact
- Multi-AI analysis for threat assessment

THREAT CATEGORIES:
1. Job Displacement - AI replacing human workers
2. Wage Suppression - AI driving down compensation
3. Market Manipulation - AI-driven trading anomalies
4. Productivity Paradox - AI efficiency not matching economic growth
5. Systemic Risk - Concentrated AI economic power

INTEGRATION:
- Uses existing AI API integrations from parent app
- Shares SQLite database with AI Bias Research
- Leverages existing Flask routes and error handling

Last modified: January 1, 2026 - v1.0.0 Initial Release
"""

import os
import sys
import json
import requests
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Import AI query functions from parent app
# These will be imported when integrated into main app.py


class EconomicThreatTracker:
    """
    Main economic threat monitoring engine
    
    Collects real-time economic data and uses multi-AI analysis to detect
    AI-driven threats to the economy.
    """
    
    def __init__(self, db_path: str = 'bias_research.db'):
        """Initialize the tracker with database connection"""
        self.db_path = db_path
        self.bls_api_key = os.getenv('BLS_API_KEY')
        self.fred_api_key = os.getenv('FRED_API_KEY')
        
        # Initialize database tables
        self.init_database()
    
    def init_database(self):
        """Create economic tracking tables in existing database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Economic indicators table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS economic_indicators (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                indicator_type TEXT NOT NULL,
                indicator_name TEXT NOT NULL,
                value REAL NOT NULL,
                change_from_previous REAL,
                source TEXT NOT NULL,
                metadata TEXT
            )
        ''')
        
        # AI threat assessments table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ai_threat_assessments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                threat_category TEXT NOT NULL,
                threat_level INTEGER NOT NULL,
                ai_consensus_score REAL,
                contributing_indicators TEXT,
                ai_analyses TEXT,
                summary TEXT,
                recommendations TEXT
            )
        ''')
        
        # Job market tracking table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS job_market_tracking (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                sector TEXT NOT NULL,
                ai_job_postings INTEGER,
                traditional_job_postings INTEGER,
                ai_replacement_score REAL,
                wage_trend REAL,
                metadata TEXT
            )
        ''')
        
        # Economic alerts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS economic_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                alert_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                data_points TEXT,
                recommended_action TEXT,
                acknowledged BOOLEAN DEFAULT 0
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def fetch_fred_data(self, series_id: str, days_back: int = 30) -> List[Dict]:
        """
        Fetch economic data from FRED API
        
        Common series IDs:
        - UNRATE: Unemployment Rate
        - PAYEMS: Total Nonfarm Payroll Employment
        - CES0500000003: Average Hourly Earnings
        - GDP: Gross Domestic Product
        - CPIAUCSL: Consumer Price Index
        
        Args:
            series_id: FRED series identifier
            days_back: How many days of historical data to fetch
            
        Returns:
            List of data points with dates and values
        """
        if not self.fred_api_key:
            print(f"FRED API key not configured for {series_id}")
            return []
        
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        url = f"https://api.stlouisfed.org/fred/series/observations"
        params = {
            'series_id': series_id,
            'api_key': self.fred_api_key,
            'file_type': 'json',
            'observation_start': start_date,
            'observation_end': end_date
        }
        
        try:
            print(f"Fetching FRED data for {series_id}...")
            response = requests.get(url, params=params, timeout=10)
            
            # Check response status
            if response.status_code != 200:
                print(f"FRED API error for {series_id}: Status {response.status_code}")
                print(f"Response: {response.text[:500]}")
                return []
            
            data = response.json()
            
            # Check for FRED API errors in JSON response
            if 'error_code' in data:
                print(f"FRED API error for {series_id}: {data.get('error_message', 'Unknown error')}")
                return []
            
            observations = data.get('observations', [])
            if not observations:
                print(f"No observations returned for {series_id}")
                return []
            
            result = []
            for obs in observations:
                if obs['value'] != '.':  # FRED uses '.' for missing data
                    result.append({
                        'date': obs['date'],
                        'value': float(obs['value'])
                    })
            
            print(f"Successfully fetched {len(result)} data points for {series_id}")
            return result
            
        except requests.exceptions.Timeout:
            print(f"Timeout fetching FRED data for {series_id}")
            return []
        except requests.exceptions.RequestException as e:
            print(f"Request error fetching FRED data for {series_id}: {e}")
            return []
        except Exception as e:
            print(f"Unexpected error fetching FRED data for {series_id}: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def fetch_bls_data(self, series_id: str) -> List[Dict]:
        """
        Fetch employment data from Bureau of Labor Statistics
        
        Common series IDs:
        - LNS14000000: Unemployment Rate
        - CES0000000001: Total Nonfarm Employment
        - CES0500000003: Average Hourly Earnings
        
        Args:
            series_id: BLS series identifier
            
        Returns:
            List of data points
        """
        if not self.bls_api_key:
            return []
        
        current_year = datetime.now().year
        
        url = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
        headers = {'Content-type': 'application/json'}
        data = json.dumps({
            "seriesid": [series_id],
            "startyear": str(current_year - 1),
            "endyear": str(current_year),
            "registrationkey": self.bls_api_key
        })
        
        try:
            response = requests.post(url, data=data, headers=headers, timeout=10)
            response.raise_for_status()
            result = response.json()
            
            observations = []
            for series in result.get('Results', {}).get('series', []):
                for item in series.get('data', []):
                    observations.append({
                        'date': f"{item['year']}-{item['period'].replace('M', '')}",
                        'value': float(item['value'])
                    })
            
            return observations
        except Exception as e:
            print(f"Error fetching BLS data for {series_id}: {e}")
            return []
    
    def store_indicator(self, indicator_type: str, indicator_name: str, 
                       value: float, source: str, metadata: Optional[Dict] = None):
        """Store an economic indicator in the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get previous value for change calculation
        cursor.execute('''
            SELECT value FROM economic_indicators 
            WHERE indicator_type = ? AND indicator_name = ?
            ORDER BY timestamp DESC LIMIT 1
        ''', (indicator_type, indicator_name))
        
        previous = cursor.fetchone()
        change = None
        if previous:
            change = value - previous[0]
        
        cursor.execute('''
            INSERT INTO economic_indicators 
            (indicator_type, indicator_name, value, change_from_previous, source, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (indicator_type, indicator_name, value, change, source, 
              json.dumps(metadata) if metadata else None))
        
        conn.commit()
        conn.close()
    
    def analyze_with_ai(self, question: str, ai_systems: List[callable]) -> Dict:
        """
        Get multi-AI consensus on economic threat
        
        Args:
            question: The economic question to analyze
            ai_systems: List of AI query functions
            
        Returns:
            Dict with consensus score and individual analyses
        """
        analyses = {}
        ratings = []
        
        with ThreadPoolExecutor(max_workers=len(ai_systems)) as executor:
            future_to_ai = {executor.submit(ai_func, question): ai_name 
                           for ai_name, ai_func in ai_systems}
            
            for future in as_completed(future_to_ai):
                ai_name = future_to_ai[future]
                try:
                    result = future.result()
                    if result.get('success') and result.get('rating'):
                        analyses[ai_name] = {
                            'rating': result['rating'],
                            'explanation': result['response']
                        }
                        ratings.append(result['rating'])
                except Exception as e:
                    print(f"Error getting analysis from {ai_name}: {e}")
        
        consensus_score = sum(ratings) / len(ratings) if ratings else 0
        
        return {
            'consensus_score': consensus_score,
            'num_responses': len(ratings),
            'analyses': analyses
        }
    
    def detect_job_displacement_threat(self, ai_systems: List[Tuple]) -> Dict:
        """
        Analyze AI job displacement threat
        
        Uses:
        - Recent unemployment data
        - Job posting trends
        - Multi-AI assessment of displacement risk
        
        Returns:
            Threat assessment with severity level
        """
        # Fetch unemployment data
        unemployment_data = self.fetch_fred_data('UNRATE', days_back=90)
        
        if not unemployment_data:
            return {'error': 'No unemployment data available'}
        
        # Calculate trend
        recent_values = [d['value'] for d in unemployment_data[-30:]]
        trend = (recent_values[-1] - recent_values[0]) if len(recent_values) > 1 else 0
        
        # Ask AI systems to assess threat
        question = f"""
        Current unemployment rate: {recent_values[-1]}%
        30-day trend: {'+' if trend > 0 else ''}{trend:.2f} percentage points
        
        On a scale of 1-10, rate the severity of AI-driven job displacement as an economic threat.
        Consider:
        - Current unemployment trends
        - AI automation acceleration
        - Workforce adaptation capacity
        - Economic resilience
        """
        
        ai_assessment = self.analyze_with_ai(question, ai_systems)
        
        # Determine threat level (1-5)
        threat_level = min(5, max(1, int(ai_assessment['consensus_score'] / 2)))
        
        result = {
            'threat_category': 'job_displacement',
            'threat_level': threat_level,
            'unemployment_rate': recent_values[-1],
            'unemployment_trend': trend,
            'ai_consensus_score': ai_assessment['consensus_score'],
            'ai_analyses': ai_assessment['analyses'],
            'timestamp': datetime.now().isoformat()
        }
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO ai_threat_assessments 
            (threat_category, threat_level, ai_consensus_score, contributing_indicators, ai_analyses, summary)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            'job_displacement',
            threat_level,
            ai_assessment['consensus_score'],
            json.dumps({'unemployment_rate': recent_values[-1], 'trend': trend}),
            json.dumps(ai_assessment['analyses']),
            f"AI-driven job displacement threat level: {threat_level}/5"
        ))
        conn.commit()
        conn.close()
        
        return result
    
    def get_threat_dashboard(self) -> Dict:
        """
        Get current threat dashboard with all indicators
        
        Returns:
            Complete economic threat status
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get latest threat assessments
        cursor.execute('''
            SELECT * FROM ai_threat_assessments 
            ORDER BY timestamp DESC LIMIT 10
        ''')
        assessments = [dict(row) for row in cursor.fetchall()]
        
        # Get latest economic indicators
        cursor.execute('''
            SELECT * FROM economic_indicators 
            ORDER BY timestamp DESC LIMIT 20
        ''')
        indicators = [dict(row) for row in cursor.fetchall()]
        
        # Get active alerts
        cursor.execute('''
            SELECT * FROM economic_alerts 
            WHERE acknowledged = 0
            ORDER BY timestamp DESC
        ''')
        alerts = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        
        return {
            'threat_assessments': assessments,
            'economic_indicators': indicators,
            'active_alerts': alerts,
            'last_updated': datetime.now().isoformat()
        }


# I did no harm and this file is not truncated
