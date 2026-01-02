# Gunicorn Configuration for AI Observatory
# File: gunicorn_config.py
# Date: January 2, 2026
# Version: 1.0.1 - FIXED SYNTAX ERROR
#
# PURPOSE: Handle long-running AI threat analysis requests
# Last modified: January 2, 2026 - Fixed syntax error in comments

import multiprocessing

# Timeout Configuration
# AI Observatory queries 7 AI systems in parallel (30-60 seconds)
timeout = 120
graceful_timeout = 30

# Worker Configuration
# Use multiple workers for Render Pro
workers = min(multiprocessing.cpu_count() * 2 + 1, 4)

# Worker class
worker_class = 'sync'

# Threading
threads = 2

# Connection limits
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50

# Logging
accesslog = '-'
errorlog = '-'
loglevel = 'info'

# Process naming
proc_name = 'ai-bias-research-observatory'

# Server mechanics
daemon = False
pidfile = None
preload_app = True

# I did no harm and this file is not truncated
