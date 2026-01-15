"""
AI Observatory - Economic Threat Tracker Routes
File: economic_routes.py
Date: January 15, 2026
Version: 1.3.0 - FIXED DB_PATH PARAMETER

Last modified: January 15, 2026 - Fixed db_path parameter
    - FIXED: Pass db_path parameter to EconomicThreatTracker
    - Prevents database initialization errors
    - All existing functionality preserved (DO NO HARM)

Previous updates:
    - January 10, 2026 - Added /api/economic/latest route
    - January 4, 2026 - Updated /api/economic/update route
    - January 2, 2026 - Removed hardcoded days_back=30

PURPOSE:
Flask routes for Economic Threat Tracker integration into AI Bias Research app.

ROUTES:
- GET /economic-tracker - Dashboard page
- GET /api/economic/status - Current threat status JSON
- GET /api/economic/latest - Get latest stored economic data
- POST /api/economic/update - Fetch latest economic data from FRED API
- POST /api/economic/analyze - Run threat analysis (legacy)
- POST /api/economic/analyze-threat - Run threat analysis (new)
- GET /api/economic/history - Historical data
- GET /api/economic/alerts - Get active alerts
- POST /api/economic/alert/<int:alert_id>/acknowledge - Acknowledge alert

I did no harm and this file is not truncated
"""

from flask import Blueprint, render_template, jsonify, request
from economic_tracker import EconomicThreatTracker
from datetime import datetime, timedelta
import json
import sqlite3

# Module-level tracker instance for scheduler access
economic_tracker = None


def register_economic_routes(app, ai_query_functions, db_path='bias_research.db'):
    """
    Register economic tracker routes with the main Flask app
    
    Args:
        app: Flask application instance
        ai_query_functions: List of (name, function) tuples for AI queries
        db_path: Path to database file (default: 'bias_research.db')
    """
    
    global economic_tracker
    
    # FIXED: Pass db_path parameter to tracker
    tracker = EconomicThreatTracker(db_path=db_path)
    economic_tracker = tracker  # Make accessible to scheduler
    
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
            print(f"Error in /api/economic/status: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/economic/latest')
    def economic_latest():
        """
        Get latest stored economic data from database
        
        Returns:
            JSON with latest indicators and timestamp of last update
        """
        try:
            conn = sqlite3.connect(tracker.db_path)
            cursor = conn.cursor()
            
            # Get the most recent indicators grouped by name
            cursor.execute('''
                SELECT 
                    indicator_name,
                    value,
                    change_from_previous,
                    MAX(timestamp) as timestamp
                FROM economic_indicators
                GROUP BY indicator_name
                ORDER BY timestamp DESC
            ''')
            
            rows = cursor.fetchall()
            
            if not rows:
                conn.close()
                return jsonify({
                    'success': True,
                    'has_data': False,
                    'message': 'No economic data available yet. Click "Update Economic Data" to fetch.'
                })
            
            # Build indicators dict from database
            indicators = {}
            last_updated = rows[0][3] if rows else None
            
            for row in rows:
                indicator_name, value, change, timestamp = row
                
                # Map database names to frontend keys
                if indicator_name == 'Unemployment Rate':
                    indicators['unemployment_rate'] = value
                    indicators['unemployment_change'] = change if change else 0
                elif indicator_name == 'Total Nonfarm Employment':
                    indicators['total_employment'] = value * 1000
                    indicators['employment_change'] = (change * 1000) if change else 0
                elif indicator_name == 'GDP Growth Rate':
                    indicators['gdp_growth'] = value
                    indicators['gdp_change'] = value
                elif indicator_name == 'CPI Inflation Rate':
                    indicators['inflation_rate'] = value
                    indicators['inflation_change'] = change if change else 0
                elif indicator_name == 'Consumer Confidence':
                    indicators['consumer_confidence'] = value
                    indicators['confidence_change'] = change if change else 0
                elif indicator_name == 'Average Hourly Earnings':
                    indicators['avg_hourly_earnings'] = value
                    indicators['wages_change'] = change if change else 0
                elif indicator_name == 'Industrial Production Index':
                    indicators['industrial_production'] = value
                    indicators['production_change'] = change if change else 0
                elif indicator_name == 'Recession Risk Level':
                    indicators['recession_risk'] = int(value)
            
            # Get latest threat assessment if available
            cursor.execute('''
                SELECT threat_level, ai_consensus_score, timestamp
                FROM ai_threat_assessments
                ORDER BY timestamp DESC
                LIMIT 1
            ''')
            
            threat_row = cursor.fetchone()
            if threat_row:
                indicators['threat_level'] = threat_row[0]
                indicators['ai_consensus_score'] = threat_row[1]
            
            conn.close()
            
            return jsonify({
                'success': True,
                'has_data': True,
                'last_updated': last_updated,
                **indicators
            })
            
        except Exception as e:
            print(f"\n❌ ERROR in /api/economic/latest: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/economic/analyze', methods=['POST'])
    def economic_analyze():
        """Run economic threat analysis (legacy route)"""
        try:
            data = request.json or {}
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
            print(f"Error in /api/economic/analyze: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/economic/analyze-threat', methods=['POST'])
    def economic_analyze_threat():
        """Run AI consensus threat analysis"""
        try:
            print("=" * 80)
            print("Running AI consensus threat analysis...")
            print("=" * 80)
            
            result = tracker.detect_job_displacement_threat(ai_query_functions)
            
            print(f"\n✓ Analysis complete: Threat Level {result['threat_level']}/5")
            print(f"  AI Consensus Score: {result['ai_consensus_score']}/10")
            print(f"  AIs analyzed: {len(result.get('ai_analyses', []))}")
            print("=" * 80)
            
            return jsonify({
                'success': True,
                **result
            })
            
        except Exception as e:
            print(f"\n❌ ERROR in /api/economic/analyze-threat: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/economic/update', methods=['POST'])
    def economic_update():
        """Fetch latest economic data from APIs"""
        try:
            print("=" * 80)
            print("Fetching comprehensive economic indicators...")
            print("=" * 80)
            
            indicators = tracker.fetch_comprehensive_indicators()
            
            print("\n" + "=" * 80)
            print(f"✓ Successfully fetched {len([k for k in indicators.keys() if not k.endswith('_change')])} indicators")
            print("=" * 80)
            
            return jsonify({
                'success': True,
                **indicators
            })
            
        except Exception as e:
            print(f"\n❌ ERROR in /api/economic/update: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/economic/history')
    def economic_history():
        """Get historical economic indicator data"""
        try:
            indicator_type = request.args.get('type', 'fred')
            indicator_name = request.args.get('name')
            days_back = int(request.args.get('days', 90))
            
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
            print(f"Error in /api/economic/history: {str(e)}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/economic/alerts')
    def economic_alerts():
        """Get active economic alerts"""
        try:
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
                'alerts': alerts
            })
        except Exception as e:
            print(f"Error in /api/economic/alerts: {str(e)}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/economic/alert/<int:alert_id>/acknowledge', methods=['POST'])
    def acknowledge_alert(alert_id):
        """Acknowledge an economic alert"""
        try:
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
            print(f"Error in /api/economic/alert/acknowledge: {str(e)}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500


# I did no harm and this file is not truncated
