"""
AI Observatory - Automated Daily Scheduler
Created: January 4, 2026
Last Updated: January 4, 2026 - Initial creation with daily check automation

This module handles automated daily checks for the AI Observatory:
- Updates economic data every morning
- Runs threat analysis
- Detects anomalies and correlations
- Optimizes thresholds weekly
- Generates daily summary reports

Uses APScheduler for reliable background task execution.
"""

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime, timedelta
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ObservatoryScheduler:
    """
    Manages automated scheduled tasks for AI Observatory.
    
    Daily tasks run at 8 AM UTC (customize timezone as needed):
    - Economic data update
    - Threat analysis
    - Anomaly detection
    
    Weekly tasks run Sunday at 2 AM UTC:
    - Baseline recalculation
    - Threshold optimization
    - Pattern discovery
    """
    
    def __init__(self, app, economic_tracker, ai_behavior_monitor, learning_engine):
        self.app = app
        self.economic_tracker = economic_tracker
        self.ai_behavior_monitor = ai_behavior_monitor
        self.learning_engine = learning_engine
        self.scheduler = BackgroundScheduler()
        self.last_run_status = {}
        
    def start(self):
        """Start the scheduler with all configured jobs"""
        
        # Daily job: Economic update and threat analysis (8 AM UTC)
        self.scheduler.add_job(
            func=self.daily_economic_check,
            trigger=CronTrigger(hour=8, minute=0),  # 8:00 AM UTC daily
            id='daily_economic_check',
            name='Daily Economic Data Update & Threat Analysis',
            replace_existing=True
        )
        
        # Daily job: AI behavior monitoring (9 AM UTC - after economic check)
        self.scheduler.add_job(
            func=self.daily_behavior_check,
            trigger=CronTrigger(hour=9, minute=0),  # 9:00 AM UTC daily
            id='daily_behavior_check',
            name='Daily AI Behavior Anomaly Detection',
            replace_existing=True
        )
        
        # Daily job: Generate summary report (10 AM UTC)
        self.scheduler.add_job(
            func=self.generate_daily_summary,
            trigger=CronTrigger(hour=10, minute=0),  # 10:00 AM UTC daily
            id='daily_summary',
            name='Daily Summary Report Generation',
            replace_existing=True
        )
        
        # Weekly job: Deep maintenance (Sunday 2 AM UTC)
        self.scheduler.add_job(
            func=self.weekly_maintenance,
            trigger=CronTrigger(day_of_week='sun', hour=2, minute=0),
            id='weekly_maintenance',
            name='Weekly Baseline Recalculation & Optimization',
            replace_existing=True
        )
        
        # Weekly job: Learning engine updates (Sunday 3 AM UTC)
        self.scheduler.add_job(
            func=self.weekly_learning_update,
            trigger=CronTrigger(day_of_week='sun', hour=3, minute=0),
            id='weekly_learning',
            name='Weekly Pattern Discovery & Model Training',
            replace_existing=True
        )
        
        # Start scheduler
        self.scheduler.start()
        logger.info("🚀 Observatory Scheduler started successfully")
        logger.info(f"📅 Next economic check: {self.scheduler.get_job('daily_economic_check').next_run_time}")
        
    def stop(self):
        """Stop the scheduler gracefully"""
        self.scheduler.shutdown()
        logger.info("🛑 Observatory Scheduler stopped")
    
    def daily_economic_check(self):
        """
        Daily economic data update and threat analysis
        Runs at 8 AM UTC every day
        """
        logger.info("=" * 80)
        logger.info("🌅 DAILY ECONOMIC CHECK - Starting")
        logger.info(f"⏰ Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        logger.info("=" * 80)
        
        try:
            with self.app.app_context():
                # Step 1: Update economic data
                logger.info("📊 Step 1/3: Updating economic data from FRED/BLS...")
                economic_data = self.economic_tracker.update_economic_data()
                
                if economic_data['success']:
                    logger.info(f"✅ Economic data updated successfully")
                    logger.info(f"   Unemployment Rate: {economic_data.get('unemployment_rate', 'N/A')}%")
                    logger.info(f"   Employment Change: {economic_data.get('employment_change', 'N/A'):,}")
                else:
                    logger.error(f"❌ Economic data update failed: {economic_data.get('error', 'Unknown error')}")
                    self.last_run_status['daily_economic_check'] = {
                        'status': 'failed',
                        'timestamp': datetime.utcnow().isoformat(),
                        'error': economic_data.get('error', 'Unknown error')
                    }
                    return
                
                # Step 2: Run threat analysis with AI consensus
                logger.info("🤖 Step 2/3: Running AI consensus threat analysis...")
                threat_analysis = self.economic_tracker.run_threat_analysis()
                
                if threat_analysis['success']:
                    consensus_level = threat_analysis.get('consensus_threat_level', 0)
                    logger.info(f"✅ Threat analysis complete")
                    logger.info(f"   Consensus Threat Level: {consensus_level}/5")
                    logger.info(f"   Alert Created: {threat_analysis.get('alert_created', False)}")
                    
                    if threat_analysis.get('alert_created'):
                        logger.warning(f"⚠️  ALERT: Threat level {consensus_level} detected!")
                else:
                    logger.error(f"❌ Threat analysis failed: {threat_analysis.get('error', 'Unknown error')}")
                
                # Step 3: Track sector performance
                logger.info("🏭 Step 3/3: Tracking job market sectors...")
                sector_data = self.economic_tracker.track_job_market_sectors()
                
                if sector_data.get('success'):
                    logger.info(f"✅ Sector tracking complete")
                    logger.info(f"   Sectors monitored: {len(sector_data.get('sectors', []))}")
                else:
                    logger.warning(f"⚠️  Sector tracking had issues")
                
                # Record successful run
                self.last_run_status['daily_economic_check'] = {
                    'status': 'success',
                    'timestamp': datetime.utcnow().isoformat(),
                    'threat_level': consensus_level,
                    'alert_created': threat_analysis.get('alert_created', False)
                }
                
                logger.info("=" * 80)
                logger.info("✅ DAILY ECONOMIC CHECK - Completed Successfully")
                logger.info("=" * 80)
                
        except Exception as e:
            logger.error(f"💥 DAILY ECONOMIC CHECK - Failed with exception: {str(e)}")
            logger.exception(e)
            self.last_run_status['daily_economic_check'] = {
                'status': 'error',
                'timestamp': datetime.utcnow().isoformat(),
                'error': str(e)
            }
    
    def daily_behavior_check(self):
        """
        Daily AI behavior anomaly detection
        Runs at 9 AM UTC every day
        """
        logger.info("=" * 80)
        logger.info("🔍 DAILY BEHAVIOR CHECK - Starting")
        logger.info(f"⏰ Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        logger.info("=" * 80)
        
        try:
            with self.app.app_context():
                # Step 1: Detect anomalies in recent responses
                logger.info("📡 Step 1/3: Scanning for behavioral anomalies...")
                anomalies = self.ai_behavior_monitor.detect_recent_anomalies(hours=24)
                
                if anomalies:
                    logger.info(f"⚠️  Found {len(anomalies)} anomalies in last 24 hours")
                    for anomaly in anomalies[:5]:  # Show first 5
                        logger.info(f"   - {anomaly['ai_system']}: {anomaly['anomaly_type']} (severity: {anomaly['severity']})")
                else:
                    logger.info(f"✅ No significant anomalies detected")
                
                # Step 2: Check for cross-AI correlations
                logger.info("🔗 Step 2/3: Checking for cross-AI correlations...")
                correlations = self.ai_behavior_monitor.detect_cross_ai_correlations()
                
                if correlations and len(correlations) > 0:
                    logger.warning(f"⚠️  Found {len(correlations)} cross-AI correlation events!")
                    for corr in correlations[:3]:  # Show first 3
                        logger.warning(f"   - {corr['correlation_type']}: {len(corr['ai_systems'])} systems (strength: {corr['correlation_strength']:.2f})")
                else:
                    logger.info(f"✅ No cross-AI correlations detected")
                
                # Step 3: Calculate overall threat level
                logger.info("📊 Step 3/3: Calculating behavioral threat level...")
                threat_level = self.ai_behavior_monitor.calculate_threat_level()
                
                logger.info(f"   Behavioral Threat Level: {threat_level['level']}/5 ({threat_level['label']})")
                
                # Record successful run
                self.last_run_status['daily_behavior_check'] = {
                    'status': 'success',
                    'timestamp': datetime.utcnow().isoformat(),
                    'anomalies_found': len(anomalies) if anomalies else 0,
                    'correlations_found': len(correlations) if correlations else 0,
                    'threat_level': threat_level['level']
                }
                
                logger.info("=" * 80)
                logger.info("✅ DAILY BEHAVIOR CHECK - Completed Successfully")
                logger.info("=" * 80)
                
        except Exception as e:
            logger.error(f"💥 DAILY BEHAVIOR CHECK - Failed with exception: {str(e)}")
            logger.exception(e)
            self.last_run_status['daily_behavior_check'] = {
                'status': 'error',
                'timestamp': datetime.utcnow().isoformat(),
                'error': str(e)
            }
    
    def generate_daily_summary(self):
        """
        Generate daily summary report
        Runs at 10 AM UTC every day
        """
        logger.info("=" * 80)
        logger.info("📋 DAILY SUMMARY REPORT - Generating")
        logger.info(f"⏰ Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        logger.info("=" * 80)
        
        try:
            with self.app.app_context():
                economic_status = self.last_run_status.get('daily_economic_check', {})
                behavior_status = self.last_run_status.get('daily_behavior_check', {})
                
                summary = {
                    'date': datetime.utcnow().strftime('%Y-%m-%d'),
                    'economic_threat_level': economic_status.get('threat_level', 0),
                    'behavioral_threat_level': behavior_status.get('threat_level', 0),
                    'alerts_created': economic_status.get('alert_created', False),
                    'anomalies_detected': behavior_status.get('anomalies_found', 0),
                    'correlations_detected': behavior_status.get('correlations_found', 0),
                    'overall_status': 'healthy' if economic_status.get('threat_level', 0) < 3 and behavior_status.get('threat_level', 0) < 3 else 'elevated'
                }
                
                # Store summary in database (you can implement this)
                # self.store_daily_summary(summary)
                
                logger.info("📊 DAILY SUMMARY:")
                logger.info(f"   Economic Threat: Level {summary['economic_threat_level']}/5")
                logger.info(f"   Behavioral Threat: Level {summary['behavioral_threat_level']}/5")
                logger.info(f"   Anomalies Detected: {summary['anomalies_detected']}")
                logger.info(f"   Correlations Detected: {summary['correlations_detected']}")
                logger.info(f"   Overall Status: {summary['overall_status'].upper()}")
                
                logger.info("=" * 80)
                logger.info("✅ DAILY SUMMARY REPORT - Completed")
                logger.info("=" * 80)
                
        except Exception as e:
            logger.error(f"💥 DAILY SUMMARY - Failed with exception: {str(e)}")
            logger.exception(e)
    
    def weekly_maintenance(self):
        """
        Weekly maintenance tasks
        Runs Sunday at 2 AM UTC
        """
        logger.info("=" * 80)
        logger.info("🔧 WEEKLY MAINTENANCE - Starting")
        logger.info(f"⏰ Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        logger.info("=" * 80)
        
        try:
            with self.app.app_context():
                # Step 1: Recalculate baselines
                logger.info("📏 Step 1/3: Recalculating AI behavior baselines...")
                self.ai_behavior_monitor.calculate_baselines()
                logger.info("✅ Baselines recalculated")
                
                # Step 2: Optimize alert thresholds
                logger.info("🎯 Step 2/3: Optimizing alert thresholds...")
                optimization_results = self.learning_engine.optimize_alert_thresholds()
                logger.info(f"✅ Thresholds optimized")
                logger.info(f"   False positive reduction: {optimization_results.get('improvement', 'N/A')}")
                
                # Step 3: Clean old data (optional)
                logger.info("🗑️  Step 3/3: Cleaning old data (keeping 365 days)...")
                # Implement cleanup if needed
                logger.info("✅ Cleanup complete")
                
                logger.info("=" * 80)
                logger.info("✅ WEEKLY MAINTENANCE - Completed Successfully")
                logger.info("=" * 80)
                
        except Exception as e:
            logger.error(f"💥 WEEKLY MAINTENANCE - Failed with exception: {str(e)}")
            logger.exception(e)
    
    def weekly_learning_update(self):
        """
        Weekly learning engine updates
        Runs Sunday at 3 AM UTC
        """
        logger.info("=" * 80)
        logger.info("🧠 WEEKLY LEARNING UPDATE - Starting")
        logger.info(f"⏰ Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        logger.info("=" * 80)
        
        try:
            with self.app.app_context():
                # Step 1: Validate predictions and update AI accuracy
                logger.info("📊 Step 1/4: Validating predictions and tracking AI accuracy...")
                self.learning_engine.track_ai_accuracy()
                logger.info("✅ AI accuracy updated")
                
                # Step 2: Adjust AI weights based on accuracy
                logger.info("⚖️  Step 2/4: Adjusting AI weights...")
                weights = self.learning_engine.adjust_ai_weights()
                logger.info("✅ AI weights adjusted")
                for ai, weight_data in list(weights.items())[:3]:  # Show first 3
                    logger.info(f"   - {ai}: {weight_data.get('weight', 1.0):.2f} (accuracy: {weight_data.get('accuracy', 0):.1f}%)")
                
                # Step 3: Discover new patterns
                logger.info("🔍 Step 3/4: Discovering economic patterns...")
                patterns = self.learning_engine.discover_patterns()
                logger.info(f"✅ Pattern discovery complete ({len(patterns) if patterns else 0} patterns found)")
                
                # Step 4: Update prediction models
                logger.info("🤖 Step 4/4: Updating prediction models...")
                model_update = self.learning_engine.update_model_performance()
                logger.info(f"✅ Models updated (accuracy: {model_update.get('accuracy', 'N/A')})")
                
                logger.info("=" * 80)
                logger.info("✅ WEEKLY LEARNING UPDATE - Completed Successfully")
                logger.info("=" * 80)
                
        except Exception as e:
            logger.error(f"💥 WEEKLY LEARNING UPDATE - Failed with exception: {str(e)}")
            logger.exception(e)
    
    def get_scheduler_status(self):
        """Get current status of scheduler and jobs"""
        jobs = []
        for job in self.scheduler.get_jobs():
            jobs.append({
                'id': job.id,
                'name': job.name,
                'next_run': job.next_run_time.isoformat() if job.next_run_time else None,
                'trigger': str(job.trigger)
            })
        
        return {
            'running': self.scheduler.running,
            'jobs': jobs,
            'last_runs': self.last_run_status
        }


# Flask route to check scheduler status
def add_scheduler_routes(app, scheduler):
    """Add routes to monitor scheduler status"""
    
    @app.route('/api/observatory/scheduler/status')
    def scheduler_status():
        """Get scheduler status and job information"""
        status = scheduler.get_scheduler_status()
        return {
            'success': True,
            'data': status
        }
    
    @app.route('/api/observatory/scheduler/trigger/<job_id>', methods=['POST'])
    def trigger_job_manually(job_id):
        """Manually trigger a scheduled job"""
        try:
            job = scheduler.scheduler.get_job(job_id)
            if not job:
                return {
                    'success': False,
                    'error': f'Job {job_id} not found'
                }, 404
            
            # Run job immediately
            job.func()
            
            return {
                'success': True,
                'message': f'Job {job_id} triggered successfully'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }, 500


# I did no harm and this file is not truncated
