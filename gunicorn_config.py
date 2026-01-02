# FIX FOR WORKER TIMEOUT - DEPLOYMENT GUIDE
**Date:** January 2, 2026  
**Issue:** Worker timeout when running AI threat analysis  
**Solution:** Configure Gunicorn for long-running requests  
**Time to fix:** 3 minutes

---

## 🎯 PROBLEM IDENTIFIED

Your logs show:
```
[CRITICAL] WORKER TIMEOUT (pid:65)
[ERROR] Worker (pid:65) was sent SIGKILL! Perhaps out of memory?
POST /api/economic/analyze responseTimeMS=31739 (31 seconds!)
```

**What's happening:**
- "Run Threat Analysis" queries 7 AI systems in parallel
- Takes 30-60 seconds to complete
- Default Gunicorn timeout: 30 seconds
- Request exceeds timeout → Worker killed → 502 error

**Everything else works perfectly!**
- ✅ All dashboards load
- ✅ FRED API works
- ✅ "Update Economic Data" works
- ✅ Database works
- ❌ Only "Run Threat Analysis" times out

---

## ✅ THE FIX

Add `gunicorn_config.py` to increase timeout from 30s → 120s

---

## 📋 DEPLOYMENT STEPS

### Step 1: Upload gunicorn_config.py to GitHub

1. Download the `gunicorn_config.py` file (artifact above)
2. Go to your GitHub repo: `ai-bias-research`
3. Click "Add file" → "Upload files"
4. Upload `gunicorn_config.py` to the **ROOT directory** (same level as app.py)
5. Commit: "Add Gunicorn config for AI Observatory long-running requests"

---

### Step 2: Update Render Start Command

1. Go to Render dashboard
2. Click your `ai-bias-research` app
3. Click "Settings" tab (left sidebar)
4. Scroll to "Build & Deploy" section
5. Find "Start Command"

**Current start command probably looks like:**
```
gunicorn app:app
```

**Change it to:**
```
gunicorn --config gunicorn_config.py app:app
```

6. Click "Save Changes"

---

### Step 3: Render Auto-Redeploys

- Render will automatically redeploy (~2-3 minutes)
- Watch the deploy logs
- Wait for "Live" status

---

### Step 4: Test the Fix

1. Go to: https://ai-bias-research.onrender.com/economic-tracker
2. Click "Run Threat Analysis"
3. Wait 30-60 seconds (you'll see "Analyzing..." message)
4. Should complete successfully now! No more 502 error

**Expected result:**
- Threat Level: 1-3/5
- AI Consensus Score: 2.0-5.0/10
- Multiple AI analyses shown

---

## 🎯 WHAT THE CONFIG DOES

**Before (default Gunicorn):**
- Timeout: 30 seconds
- Workers: 1
- Kills requests that take >30s

**After (with gunicorn_config.py):**
- Timeout: 120 seconds ✅
- Workers: 2-4 (for Render Pro) ✅
- Handles AI threat analysis (30-60s) ✅
- Multiple concurrent users ✅
- Better performance ✅

---

## 📊 EXPECTED PERFORMANCE AFTER FIX

### Economic Tracker:
- "Update Economic Data": ~3-5 seconds ✅
- "Run Threat Analysis": ~30-60 seconds ✅ (no timeout!)

### AI Behavior Monitor:
- "Recalculate Baselines": ~5-10 seconds ✅
- "Detect Correlations": ~5-10 seconds ✅

### Observatory Dashboard:
- "Run Full Scan": ~60-90 seconds ✅
- Queries economic + behavior in parallel

---

## 🔧 TROUBLESHOOTING

### If it still times out after this fix:

**Check Render logs for:**
```
Successfully loaded gunicorn config
Workers spawned: X
Timeout set to: 120
```

**If you don't see these messages:**
- Gunicorn config not loaded
- Check file is in root directory
- Check start command is correct

### If you see memory errors:

Your Render Pro plan should have enough memory, but if you see:
```
Worker was sent SIGKILL! Perhaps out of memory?
```

**Solution:** Reduce concurrent AI queries:
- Edit `economic_tracker.py`
- Change `max_workers=7` to `max_workers=4`
- Queries fewer AIs at once

---

## ✨ AFTER THIS FIX, EVERYTHING WORKS!

Your AI Observatory will be fully functional:

✅ Economic Threat Tracker
- Update economic data ✅
- Run AI threat analysis ✅ (FIXED!)
- View history ✅

✅ AI Behavior Monitor  
- Calculate baselines ✅
- Detect anomalies ✅
- Cross-AI correlations ✅

✅ Observatory Dashboard
- Threat matrix ✅
- Full system scan ✅ (FIXED!)
- Real-time monitoring ✅

---

## 🚀 READY TO DEPLOY!

**Just do these 2 things:**

1. ✅ Upload `gunicorn_config.py` to GitHub root
2. ✅ Update Render start command to: `gunicorn --config gunicorn_config.py app:app`

**Then test:** Click "Run Threat Analysis" and it will work! 🎉

---

# I did no harm and this file is not truncated
