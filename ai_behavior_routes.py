"""
AI Observatory - AI Behavior Monitor Routes
File: ai_behavior_routes.py
Date: January 15, 2026
Version: 1.1.0

PURPOSE:
Flask routes for AI Behavior Monitoring integration

ROUTES:
- GET /ai-behavior - Dashboard page
- GET /api/ai-behavior/status - Current behavior status
- POST /api/ai-behavior/analyze - Analyze AI response
- GET /api/ai-behavior/anomalies - Recent anomalies
- GET /api/ai-behavior/alerts - Get active alerts (NEW v1.1.0)
- GET /api/ai-behavior/correlations - Cross-AI correlation events
- GET /api/ai-behavior/emergent-capabilities - Get emergent capabilities
- POST /api/ai-behavior/recalculate-baselines - Recalculate all baselines
- GET /api/ai-behavior/summary - Get summary statistics

Last modified: January 15, 2026 - v1.1.0
    - ADDED /api/ai-behavior/alerts endpoint (was causing 404 errors)
    - Fixes Observatory dashboard alert loading

Previous: January 1, 2026 - v1.0.0 - Initial version
"""

from flask import Blueprint, render_template, jsonify, request
from ai_behavior_monitor import AIBehaviorMonitor
from datetime import datetime, timedelta
import json

def register_ai_behavior_routes(app):
    """
    Register AI behavior monitoring routes
    
    Args:
        app: Flask application instance
    """
    
    monitor = AIBehaviorMonitor()
    
    @app.route('/ai-behavior')
    def ai_behavior_dashboard():
        """AI Behavior Monitor dashboard"""
        return render_template('ai_behavior_monitor.html')
    
    @app.route('/api/ai-behavior/status')
    def ai_behavior_status():
        """Get current AI behavior status"""
        try:
            dashboard_data = monitor.get_behavior_dashboard()
            return jsonify({
                'success': True,
                'data': dashboard_data
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/ai-behavior/analyze', methods=['POST'])
    def analyze_ai_response():
        """Analyze a single AI response for anomalies"""
        try:
            data = request.json
            ai_system = data.get('ai_system')
            response_text = data.get('response_text')
            rating = data.get('rating')
            
            if not ai_system or not response_text:
                return jsonify({
                    'success': False,
                    'error': 'ai_system and response_text required'
                }), 400
            
            result = monitor.analyze_response(ai_system, response_text, rating)
            
            return jsonify({
                'success': True,
                'data': result
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/ai-behavior/anomalies')
    def get_anomalies():
        """Get recent anomalies"""
        try:
            days = int(request.args.get('days', 7))
            ai_system = request.args.get('ai_system')
            
            import sqlite3
            conn = sqlite3.connect(monitor.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if ai_system:
                cursor.execute('''
                    SELECT * FROM ai_anomalies
                    WHERE ai_system = ?
                    AND detected_at > datetime('now', '-' || ? || ' days')
                    ORDER BY detected_at DESC
                ''', (ai_system, days))
            else:
                cursor.execute('''
                    SELECT * FROM ai_anomalies
                    WHERE detected_at > datetime('now', '-' || ? || ' days')
                    ORDER BY severity DESC, detected_at DESC
                    LIMIT 100
                ''', (days,))
            
            anomalies = [dict(row) for row in cursor.fetchall()]
            conn.close()
            
            return jsonify({
                'success': True,
                'data': anomalies
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/ai-behavior/alerts')
    def get_ai_behavior_alerts():
        """
        Get active AI behavior alerts
        NEW in v1.1.0 - Was missing, causing 404 errors in Observatory dashboard
        
        Returns high-severity anomalies from the last 7 days
        """
        try:
            import sqlite3
            conn = sqlite3.connect(monitor.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get recent high-severity anomalies as alerts
            cursor.execute('''
                SELECT 
                    id,
                    ai_system,
                    anomaly_type as alert_type,
                    description,
                    severity,
                    detected_at,
                    metadata
                FROM ai_anomalies
                WHERE severity >= 3
                AND detected_at > datetime('now', '-7 days')
                ORDER BY severity DESC, detected_at DESC
                LIMIT 50
            ''')
            
            alerts = [dict(row) for row in cursor.fetchall()]
            conn.close()
            
            return jsonify({
                'success': True,
                'alerts': alerts
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e),
                'alerts': []
            }), 500
    
    @app.route('/api/ai-behavior/correlations')
    def get_correlations():
        """Get cross-AI correlation events"""
        try:
            time_window = int(request.args.get('hours', 24))
            
            correlations = monitor.detect_cross_ai_correlation(time_window)
            
            return jsonify({
                'success': True,
                'data': correlations
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/ai-behavior/emergent-capabilities')
    def get_emergent_capabilities():
        """Get detected emergent capabilities"""
        try:
            import sqlite3
            conn = sqlite3.connect(monitor.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM emergent_capabilities
                ORDER BY risk_level DESC, confirmation_count DESC
            ''')
            
            capabilities = [dict(row) for row in cursor.fetchall()]
            conn.close()
            
            return jsonify({
                'success': True,
                'data': capabilities
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/ai-behavior/recalculate-baselines', methods=['POST'])
    def recalculate_baselines():
        """Recalculate all AI baselines from historical data"""
        try:
            monitor.calculate_baselines()
            
            return jsonify({
                'success': True,
                'message': 'Baselines recalculated successfully',
                'baselines_count': len(monitor.baselines)
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/ai-behavior/summary')
    def get_behavior_summary():
        """Get summary statistics"""
        try:
            days = int(request.args.get('days', 30))
            summary = monitor.get_anomaly_summary(days)
            
            return jsonify({
                'success': True,
                'data': summary
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500


# I did no harm and this file is not truncated
