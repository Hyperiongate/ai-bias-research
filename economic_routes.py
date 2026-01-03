"""
AI Observatory - Economic Threat Tracker Routes
File: economic_routes.py
Date: January 2, 2026
Version: 1.0.1 - FIXED HARDCODED days_back BUG

Last modified: January 2, 2026 - Removed hardcoded days_back=30 to use default 730

PURPOSE:
Flask routes for Economic Threat Tracker integration into AI Bias Research app.

CRITICAL FIX:
Changed `tracker.fetch_fred_data(indicator_id, days_back=30)` 
     to `tracker.fetch_fred_data(indicator_id)`
to use the default days_back=730 from economic_tracker.py

ROUTES:
- GET /economic-tracker - Dashboard page
- GET /api/economic/status - Current threat status JSON
- POST /api/economic/analyze - Run threat analysis
- GET /api/economic/history - Historical data
- POST /api/economic/update - Fetch latest economic data

INTEGRATION:
Add these routes to your existing app.py:
    from economic_routes import register_economic_routes
    register_economic_routes(app, ai_query_functions)

I did no harm and this file is not truncated
"""

from flask import Blueprint, render_template, jsonify, request
from economic_tracker import EconomicThreatTracker
from datetime import datetime, timedelta
import json

# Create blueprint
economic_bp = Blueprint('economic', __name__, url_prefix='/economic-tracker')


def register_economic_routes(app, ai_query_functions):
    """
    Register economic tracker routes with the main Flask app
    
    Args:
        app: Flask application instance
        ai_query_functions: List of (name, function) tuples for AI queries
    """
    
    tracker = EconomicThreatTracker()
    
    @app.route('/economic-tracker')
    def economic_dashboard():
        """Main economic threat tracker dashboard"""
        return render_template('economic_tracker.html')
    
    @app.route('/api/economic/status')
    def economic_status():
        """Get current economic threat status"""
        try:
            dashboard_data = tracker.get_threat_dashboard()
            return jsonify({
                'success': True,
                'data': dashboard_data
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/economic/analyze', methods=['POST'])
    def economic_analyze():
        """Run economic threat analysis"""
        try:
            # Get threat type from request
            data = request.json
            threat_type = data.get('threat_type', 'job_displacement')
            
            if threat_type == 'job_displacement':
                result = tracker.detect_job_displacement_threat(ai_query_functions)
            else:
                return jsonify({
                    'success': False,
                    'error': f'Unknown threat type: {threat_type}'
                }), 400
            
            return jsonify({
                'success': True,
                'data': result
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/economic/update', methods=['POST'])
    def economic_update():
        """Fetch latest economic data from APIs"""
        try:
            data = request.json
            indicators = data.get('indicators', ['UNRATE', 'PAYEMS', 'CES0500000003'])
            
            results = {}
            
            for indicator_id in indicators:
                # CRITICAL FIX: Don't override days_back! Use default from economic_tracker.py (730 days)
                fred_data = tracker.fetch_fred_data(indicator_id)
                
                if fred_data:
                    latest = fred_data[-1]
                    
                    # Map indicator IDs to readable names
                    indicator_names = {
                        'UNRATE': 'Unemployment Rate',
                        'PAYEMS': 'Total Employment',
                        'CES0500000003': 'Average Hourly Earnings',
                        'GDP': 'GDP',
                        'CPIAUCSL': 'Consumer Price Index'
                    }
                    
                    tracker.store_indicator(
                        indicator_type='fred',
                        indicator_name=indicator_names.get(indicator_id, indicator_id),
                        value=latest['value'],
                        source='FRED API',
                        metadata={'series_id': indicator_id, 'date': latest['date']}
                    )
                    
                    results[indicator_id] = {
                        'success': True,
                        'value': latest['value'],
                        'date': latest['date']
                    }
                else:
                    results[indicator_id] = {
                        'success': False,
                        'error': 'No data available'
                    }
            
            return jsonify({
                'success': True,
                'data': results
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/economic/history')
    def economic_history():
        """Get historical economic indicator data"""
        try:
            # Get query parameters
            indicator_type = request.args.get('type', 'fred')
            indicator_name = request.args.get('name')
            days_back = int(request.args.get('days', 90))
            
            import sqlite3
            conn = sqlite3.connect(tracker.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if indicator_name:
                cursor.execute('''
                    SELECT * FROM economic_indicators
                    WHERE indicator_type = ? AND indicator_name = ?
                    AND timestamp > datetime('now', '-' || ? || ' days')
                    ORDER BY timestamp ASC
                ''', (indicator_type, indicator_name, days_back))
            else:
                cursor.execute('''
                    SELECT * FROM economic_indicators
                    WHERE indicator_type = ?
                    AND timestamp > datetime('now', '-' || ? || ' days')
                    ORDER BY timestamp ASC
                ''', (indicator_type, days_back))
            
            history = [dict(row) for row in cursor.fetchall()]
            conn.close()
            
            return jsonify({
                'success': True,
                'data': history
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/economic/alerts')
    def economic_alerts():
        """Get active economic alerts"""
        try:
            import sqlite3
            conn = sqlite3.connect(tracker.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM economic_alerts
                WHERE acknowledged = 0
                ORDER BY severity DESC, timestamp DESC
            ''')
            
            alerts = [dict(row) for row in cursor.fetchall()]
            conn.close()
            
            return jsonify({
                'success': True,
                'data': alerts
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/economic/alert/<int:alert_id>/acknowledge', methods=['POST'])
    def acknowledge_alert(alert_id):
        """Acknowledge an economic alert"""
        try:
            import sqlite3
            conn = sqlite3.connect(tracker.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE economic_alerts
                SET acknowledged = 1
                WHERE id = ?
            ''', (alert_id,))
            
            conn.commit()
            conn.close()
            
            return jsonify({
                'success': True,
                'message': f'Alert {alert_id} acknowledged'
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500


# I did no harm and this file is not truncated
