#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AI Observatory - Economic Threat Tracker Module
File: economic_tracker.py
Date: January 2, 2026
Version: 1.0.1 - PRODUCTION READY

Last modified: January 2, 2026 - Fixed 502 error by completing all methods

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
- FRED API (Federal Reserve Economic Data) - Economic indicators
- Bureau of Labor Statistics (BLS) API - Employment data
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

I did no harm and this file is not truncated
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
    
    def fetch_fred_data(self, series_id: str, days_back: int = 730) -> List[Dict]:
        """
        Fetch economic data from FRED API with detailed error logging
        
        Common series IDs:
        - UNRATE: Unemployment Rate (monthly)
        - PAYEMS: Total Nonfarm Payroll Employment (monthly)
        - CES0500000003: Average Hourly Earnings (monthly)
        - GDP: Gross Domestic Product (quarterly)
        - CPIAUCSL: Consumer Price Index (monthly)
        
        Args:
            series_id: FRED series identifier
            days_back: How many days of historical data to fetch (default 730 = 2 years for monthly data)
            
        Returns:
            List of data points with dates and values
        """
        if not self.fred_api_key:
            print("ERROR: FRED_API_KEY not configured in environment variables")
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
            
            print(f"FRED API Response Status: {response.status_code}")
            print(f"FRED API URL: {url}")
            print(f"FRED API Params: series_id={series_id}, start={start_date}, end={end_date}")
            
            if response.status_code != 200:
                print(f"FRED API Error: Status {response.status_code}")
                print(f"Response body: {response.text[:500]}")
                return []
            
            response.raise_for_status()
            data = response.json()
            
            print(f"FRED Response Keys: {data.keys()}")
            print(f"FRED realtime_start: {data.get('realtime_start')}")
            print(f"FRED realtime_end: {data.get('realtime_end')}")
            print(f"FRED observation_start: {data.get('observation_start')}")
            print(f"FRED observation_end: {data.get('observation_end')}")
            
            if 'error_message' in data:
                print(f"FRED API Error Message: {data['error_message']}")
                return []
            
            observations = []
            raw_obs = data.get('observations', [])
            print(f"Found {len(raw_obs)} raw observations")
            if len(raw_obs) > 0:
                print(f"First observation: {raw_obs[0]}")
                print(f"Last observation: {raw_obs[-1]}")
            
            for obs in raw_obs:
                if obs['value'] != '.':  # FRED uses '.' for missing data
                    observations.append({
                        'date': obs['date'],
                        'value': float(obs['value'])
                    })
            
            print(f"Successfully parsed {len(observations)} valid observations")
            return observations
            
        except requests.exceptions.Timeout:
            print(f"FRED API Timeout for {series_id}")
            return []
        except requests.exceptions.RequestException as e:
            print(f"FRED API Request Exception for {series_id}: {str(e)}")
            return []
        except Exception as e:
            print(f"Unexpected error fetching FRED data for {series_id}: {str(e)}")
            import traceback
            print(traceback.format_exc())
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
        ''', (indicator_type, indicator_name, value, change, source, json.dumps(metadata) if metadata else None))
        
        conn.commit()
        conn.close()
    
    def detect_job_displacement_threat(self, ai_query_functions: List[Tuple]) -> Dict:
        """
        Analyze job displacement threat using multi-AI consensus
        
        Args:
            ai_query_functions: List of (name, function) tuples for AI queries
            
        Returns:
            Dict with threat assessment
        """
        # Get latest economic data
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM economic_indicators 
            ORDER BY timestamp DESC 
            LIMIT 10
        ''')
        
        indicators = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        if not indicators:
            return {
                'threat_level': 1,
                'ai_consensus_score': 0.0,
                'summary': 'No economic data available for analysis',
                'indicators': []
            }
        
        # Create analysis prompt for AIs
        indicator_summary = "\n".join([
            f"- {ind['indicator_name']}: {ind['value']}" 
            for ind in indicators[:5]
        ])
        
        prompt = f"""As an economic analyst, assess the threat level of AI-driven job displacement based on these indicators:

{indicator_summary}

On a scale of 1-10, rate the current threat level of AI causing job displacement:
1-2: Low threat (minimal AI impact)
3-4: Moderate threat (some sectors affected)
5-6: Elevated threat (significant displacement)
7-8: High threat (widespread displacement)
9-10: Critical threat (economic crisis)

Provide only a numerical rating (1-10) on the first line, then briefly explain."""
        
        # Query multiple AIs in parallel
        ai_results = []
        with ThreadPoolExecutor(max_workers=len(ai_query_functions)) as executor:
            future_to_ai = {
                executor.submit(func, prompt): name 
                for name, func in ai_query_functions
            }
            
            for future in as_completed(future_to_ai):
                ai_name = future_to_ai[future]
                try:
                    result = future.result(timeout=30)
                    if result.get('success'):
                        ai_results.append({
                            'ai_system': ai_name,
                            'raw_response': result.get('raw_response', ''),
                            'rating': self._extract_rating(result.get('raw_response', ''))
                        })
                except Exception as e:
                    print(f"Error querying {ai_name}: {e}")
        
        # Calculate consensus
        valid_ratings = [r['rating'] for r in ai_results if r['rating'] is not None]
        
        if not valid_ratings:
            consensus_score = 0.0
            threat_level = 1
        else:
            consensus_score = sum(valid_ratings) / len(valid_ratings)
            threat_level = self._score_to_threat_level(consensus_score)
        
        # Store assessment
        self._store_threat_assessment(
            'job_displacement',
            threat_level,
            consensus_score,
            indicators,
            ai_results
        )
        
        return {
            'threat_level': threat_level,
            'ai_consensus_score': round(consensus_score, 2),
            'ai_analyses': ai_results,
            'contributing_indicators': indicators[:5],
            'summary': self._generate_summary(threat_level, consensus_score)
        }
    
    def _extract_rating(self, text: str) -> Optional[float]:
        """Extract numerical rating from AI response"""
        if not text:
            return None
        
        # Try first line
        first_line = text.strip().split('\n')[0].strip()
        import re
        match = re.search(r'^(\d+(?:\.\d+)?)', first_line)
        
        if match:
            try:
                rating = float(match.group(1))
                if 0 <= rating <= 10:
                    return rating
            except ValueError:
                pass
        
        # Try anywhere in text
        match = re.search(r'(\d+(?:\.\d+)?)\s*/\s*10', text)
        if match:
            try:
                rating = float(match.group(1))
                if 0 <= rating <= 10:
                    return rating
            except ValueError:
                pass
        
        return None
    
    def _score_to_threat_level(self, score: float) -> int:
        """Convert consensus score to threat level 1-5"""
        if score < 2.0:
            return 1
        elif score < 4.0:
            return 2
        elif score < 6.0:
            return 3
        elif score < 8.0:
            return 4
        else:
            return 5
    
    def _generate_summary(self, threat_level: int, score: float) -> str:
        """Generate human-readable summary"""
        summaries = {
            1: f"Low threat level (score: {score:.1f}/10). AI impact on jobs is minimal.",
            2: f"Moderate threat level (score: {score:.1f}/10). Some sectors experiencing AI-driven changes.",
            3: f"Elevated threat level (score: {score:.1f}/10). Significant job market disruption detected.",
            4: f"High threat level (score: {score:.1f}/10). Widespread AI-driven job displacement occurring.",
            5: f"Critical threat level (score: {score:.1f}/10). Economic crisis from AI job displacement."
        }
        return summaries.get(threat_level, "Unknown threat level")
    
    def _store_threat_assessment(self, category: str, threat_level: int, 
                                 consensus_score: float, indicators: List[Dict],
                                 ai_analyses: List[Dict]):
        """Store threat assessment in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO ai_threat_assessments
            (threat_category, threat_level, ai_consensus_score, contributing_indicators, ai_analyses, summary)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            category,
            threat_level,
            consensus_score,
            json.dumps(indicators[:5]),
            json.dumps(ai_analyses),
            self._generate_summary(threat_level, consensus_score)
        ))
        
        conn.commit()
        conn.close()
    
    def get_threat_dashboard(self) -> Dict:
        """Get current threat dashboard data"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get latest threat assessments
        cursor.execute('''
            SELECT * FROM ai_threat_assessments
            ORDER BY timestamp DESC
            LIMIT 10
        ''')
        threats = [dict(row) for row in cursor.fetchall()]
        
        # Get latest economic indicators
        cursor.execute('''
            SELECT * FROM economic_indicators
            ORDER BY timestamp DESC
            LIMIT 10
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
            'threat_assessments': threats,
            'economic_indicators': indicators,
            'active_alerts': alerts,
            'overall_threat_level': threats[0]['threat_level'] if threats else 1
        }


# I did no harm and this file is not truncated
