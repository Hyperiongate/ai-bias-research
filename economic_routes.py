"""
AI Observatory - Economic Threat Tracker Routes
File: economic_routes.py
Date: January 10, 2026
Version: 1.2.0 - ADDED LATEST DATA RETRIEVAL FOR PAGE PERSISTENCE

Last modified: January 10, 2026 - Added /api/economic/latest route
    - NEW: /api/economic/latest endpoint loads previously stored data
    - Fixes blank page on revisit - shows last known values
    - Displays timestamp of when data was last updated
    - All existing functionality preserved (DO NO HARM)

Previous updates:
    - January 4, 2026 - Updated /api/economic/update route
    - Changed to use fetch_comprehensive_indicators() method
    - Now returns all 8 indicators in one call
    - Added /api/economic/analyze-threat route for clearer naming
    - January 2, 2026 - Removed hardcoded days_back=30 to use default 730

PURPOSE:
Flask routes for Economic Threat Tracker integration into AI Bias Research app.

ROUTES:
- GET /economic-tracker - Dashboard page
- GET /api/economic/status - Current threat status JSON
- GET /api/economic/latest - Get latest stored economic data (NEW v1.2.0)
- POST /api/economic/update - Fetch latest economic data from FRED API
- POST /api/economic/analyze - Run threat analysis (legacy)
- POST /api/economic/analyze-threat - Run threat analysis (new)
- GET /api/economic/history - Historical data
- GET /api/economic/alerts - Get active alerts
- POST /api/economic/alert/<int:alert_id>/acknowledge - Acknowledge alert

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
import sqlite3

# Create blueprint
economic_bp = Blueprint('economic', __name__, url_prefix='/economic-tracker')

# Module-level tracker instance for scheduler access
economic_tracker = None


def register_economic_routes(app, ai_query_functions):
    """
    Register economic tracker routes with the main Flask app
    
    Args:
        app: Flask application instance
        ai_query_functions: List of (name, function) tuples for AI queries
    """
    
    global economic_tracker
    tracker = EconomicThreatTracker()
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
        
        NEW in v1.2.0: This route loads previously stored data so the page
        shows the last known values on revisit instead of being blank.
        
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
                    indicators['total_employment'] = value * 1000  # PAYEMS is in thousands
                    indicators['employment_change'] = (change * 1000) if change else 0
                elif indicator_name == 'GDP Growth Rate':
                    indicators['gdp_growth'] = value
                    indicators['gdp_change'] = value  # Growth rate itself
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
            # Get threat type from request
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
        """
        Run AI consensus threat analysis (new clearer route name)
        
        This is the main analysis route that:
        1. Gets latest economic indicators from database
        2. Sends to 7 AI systems for analysis
        3. Calculates weighted consensus score
        4. Generates alerts if threat level >= 3
        5. Returns complete threat assessment
        """
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
                **result  # Spread all result fields into response
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
        """
        Fetch latest economic data from APIs
        
        UPDATED in v1.1.0: Now uses fetch_comprehensive_indicators() to get
        all 8 economic indicators in one efficient call instead of manually
        fetching each one.
        
        Returns:
            JSON with all indicators:
            - unemployment_rate & unemployment_change
            - total_employment & employment_change
            - gdp_growth & gdp_change
            - inflation_rate & inflation_change
            - consumer_confidence & confidence_change
            - avg_hourly_earnings & wages_change
            - industrial_production & production_change
            - recession_risk (calculated 1-5)
        """
        try:
            print("=" * 80)
            print("Fetching comprehensive economic indicators...")
            print("=" * 80)
            
            # Use the new comprehensive fetch method
            indicators = tracker.fetch_comprehensive_indicators()
            
            print("\n" + "=" * 80)
            print(f"✓ Successfully fetched {len([k for k in indicators.keys() if not k.endswith('_change')])} indicators")
            print("=" * 80)
            
            return jsonify({
                'success': True,
                **indicators  # Spread all indicators into the response
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
            print(f"Error in /api/economic/history: {str(e)}")
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
            print(f"Error in /api/economic/alert/acknowledge: {str(e)}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500


# I did no harm and this file is not truncated
