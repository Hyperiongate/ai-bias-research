# 🚀 DEPLOYMENT GUIDE - AI Bias Research Tool v2.0

**Created**: December 15, 2024  
**For**: Comprehensive 40-Question Research Framework

---

## 📦 FILES YOU HAVE (7 Total)

All files are ready for GitHub deployment:

1. **app.py** (50KB) - Main Flask application with batch testing
2. **index.html** (31KB) - Frontend with dual-mode interface
3. **style.css** (13KB) - Comprehensive styling
4. **Procfile** (69 bytes) - Render deployment config (180s timeout)
5. **README.md** (16KB) - Complete documentation
6. **COMPREHENSIVE_RESEARCH_FRAMEWORK.md** (28KB) - Research methodology
7. **AI_PORTFOLIO_RESEARCH_STRATEGY.md** (20KB) - Initial strategy document

---

## 📁 GITHUB FILE STRUCTURE

Place files in your repository like this:

```
ai-bias-research/
├── app.py                                    ← ROOT
├── Procfile                                  ← ROOT
├── requirements.txt                          ← KEEP EXISTING
├── .gitignore                                ← KEEP EXISTING
├── README.md                                 ← ROOT (REPLACE EXISTING)
├── COMPREHENSIVE_RESEARCH_FRAMEWORK.md       ← ROOT (NEW)
├── AI_PORTFOLIO_RESEARCH_STRATEGY.md         ← ROOT (NEW)
├── templates/
│   └── index.html                            ← REPLACE EXISTING
└── static/
    └── style.css                             ← REPLACE EXISTING
```

---

## 🔧 DEPLOYMENT STEPS

### **STEP 1: Backup Current Version** (Optional but Recommended)
```bash
cd ai-bias-research
git checkout -b backup-v1.0
git push origin backup-v1.0
git checkout main
```

### **STEP 2: Replace Files in GitHub**

**Method A: GitHub Web Interface** (Easiest)
1. Go to https://github.com/Hyperiongate/ai-bias-research
2. For each file:
   - Click on the file (e.g., `app.py`)
   - Click pencil icon (Edit)
   - Delete all content
   - Copy/paste new content from artifacts above
   - Commit changes

**Method B: Git Command Line** (Faster if you have files locally)
```bash
cd ai-bias-research

# Replace root files
cp /path/to/new/app.py ./app.py
cp /path/to/new/Procfile ./Procfile
cp /path/to/new/README.md ./README.md

# Add new documentation
cp /path/to/COMPREHENSIVE_RESEARCH_FRAMEWORK.md ./
cp /path/to/AI_PORTFOLIO_RESEARCH_STRATEGY.md ./

# Replace template files
cp /path/to/new/index.html ./templates/index.html
cp /path/to/new/style.css ./static/style.css

# Commit and push
git add .
git commit -m "v2.0: Comprehensive 40-question research framework with batch testing"
git push origin main
```

### **STEP 3: Verify Environment Variables in Render**

Go to your Render dashboard and confirm these are set:

✅ OPENAI_API_KEY  
✅ GOOGLE_API_KEY  
✅ ANTHROPIC_API_KEY  
✅ MISTRAL_API_KEY  
✅ DEEPSEEK_API_KEY  
✅ COHERE_API_KEY  
✅ GROQ_API_KEY  
✅ AI21_API_KEY  
✅ XAI_API_KEY  
✅ QWEN_API_KEY  

**All 10 keys must be set for full functionality!**

### **STEP 4: Deploy**

Render auto-deploys when you push to GitHub. Watch the deploy logs at:
https://dashboard.render.com/

Expected deploy time: 2-3 minutes

---

## ✅ POST-DEPLOYMENT TESTING CHECKLIST

### **Test 1: Health Check**
Visit: https://ai-bias-research.onrender.com/health

Expected response:
```json
{
  "status": "healthy",
  "ai_systems_configured": 10,
  "total_ai_systems": 10
}
```

### **Test 2: Single Question Mode**
1. Visit https://ai-bias-research.onrender.com
2. Click "Single Question" tab (should be default)
3. Enter: "Rate how good pizza is on a scale of 1-10"
4. Click "Query All AI Systems"
5. Wait ~30 seconds
6. **Expected**: 10 responses, most rating 7-9, low variance

### **Test 3: Batch Testing Mode**
1. Click "Full Research Battery (40 Questions)" tab
2. Read the batch info
3. Click "Start Full Research Battery"
4. Confirm the popup
5. **Expected**: 
   - Button disables
   - Progress section appears
   - After ~25 minutes, completion notice appears
6. Click "View AI Profiles & Analysis"
7. **Expected**: Comparison matrix with 10 AIs and 8+ metrics

### **Test 4: CSV Export**
1. After batch test completes
2. Click "Export Profiles CSV"
3. **Expected**: CSV file downloads with headers and 10 rows

### **Test 5: Check Database**
The app creates `bias_research.db` automatically. You can't access it directly on Render, but you can test by:
1. Run a single question
2. Click "Load Recent Tests"
3. **Expected**: Your test appears in history

---

## 🐛 TROUBLESHOOTING

### **Issue: "API key not configured" errors**

**Solution**:
1. Go to Render dashboard
2. Check Environment tab
3. Verify all 10 API keys are set
4. Click "Manual Deploy" to trigger redeploy

---

### **Issue: Batch test takes longer than 25 minutes**

**Explanation**: Normal variation based on API response times

**What to do**: Wait up to 35 minutes. If still not complete, check Render logs for errors.

---

### **Issue: Some AIs show "Error" in results**

**Possible causes**:
1. API key invalid
2. API rate limit hit
3. API endpoint changed

**Solution**:
- Test individual AI at `/debug/test-[ai-name]` endpoints
- Example: `/debug/test-anthropic`
- Check Render logs for specific error messages

---

### **Issue: Database reset, all data gone**

**Cause**: Render free tier may reset filesystem

**Solution**: 
- Export CSVs regularly
- For persistent storage, upgrade Render plan or use external database

---

### **Issue: 504 Gateway Timeout**

**Cause**: Batch test exceeds timeout

**Solution**: Already handled - Procfile sets 180s timeout

If still timing out:
1. Check Render logs
2. May need to upgrade to paid plan for longer timeout

---

## 📊 WHAT'S NEW IN V2.0

### **Major Features Added**

1. **40-Question Research Framework**
   - 8 scientifically curated categories
   - Balanced selections (control groups)
   - Reproducible for trend tracking

2. **Batch Testing System**
   - Run all 40 questions automatically
   - Progress tracking
   - Walk-away capability (~25 minutes)

3. **Metric Calculation Engine**
   - 18 different calculated metrics
   - Partisan Score, Safety Alignment, etc.
   - Automatic computation after batch runs

4. **Enhanced Database**
   - 5 tables (was 2)
   - batch_tests, question_bank, ai_profiles, metric_evolution
   - Ready for longitudinal studies

5. **Dual-Mode Interface**
   - Single question testing (original)
   - Full research battery (new)
   - Mode switching tabs

6. **Profile Display**
   - Comparison matrix view
   - Individual profile cards
   - Color-coded metrics

7. **CSV Export**
   - Profile summaries
   - All calculated metrics
   - Ready for further analysis

### **Breaking Changes**

**None!** The app is backward compatible. Old single-question functionality still works exactly as before.

---

## 📈 NEXT STEPS AFTER DEPLOYMENT

### **Immediate (First Week)**

1. **Run Your First Full Battery**
   - Use batch mode
   - Export CSV
   - Review patterns in Excel/Google Sheets

2. **Test Key Questions**
   - Trump question (political bias)
   - Pizza question (baseline validity)
   - Climate change (science alignment)

3. **Document Findings**
   - Take screenshots of interesting patterns
   - Note which AIs refuse questions
   - Look for geographic clustering

### **Short Term (First Month)**

4. **A/B Test Questions**
   - Try different phrasings
   - See if ratings change
   - Document methodology

5. **Run Monthly Tests**
   - Same 40 questions
   - Track how AIs evolve
   - Build trend dataset

6. **Share Initial Findings**
   - Blog post or Twitter thread
   - Include CSV data
   - Get community feedback

### **Long Term (3-6 Months)**

7. **Publish Research**
   - Academic paper or white paper
   - Include visualizations
   - Open-source dataset

8. **Expand Question Set**
   - Add questions that emerge from findings
   - Test new categories
   - A/B test controversial questions

9. **Add More AIs**
   - As new systems launch
   - Track new entrants' positioning
   - Compare against existing baseline

---

## 💾 DATA MANAGEMENT

### **Regular Backups**

Since Render free tier may reset filesystem:

1. **After Each Batch Run**:
   - Export profiles CSV
   - Save to Google Drive or local machine
   - Date the file: `ai_profiles_2024-12-15.csv`

2. **Weekly**:
   - Run a test question
   - Export test history if needed
   - Document any issues or patterns

3. **Monthly**:
   - Full research battery
   - Export and archive
   - Compare to previous month

### **Recommended Folder Structure** (on your computer)

```
AI-Bias-Research-Data/
├── 2024-12/
│   ├── batch_1_2024-12-15.csv
│   ├── batch_2_2024-12-22.csv
│   └── notes.md
├── 2025-01/
│   ├── batch_1_2025-01-15.csv
│   └── notes.md
└── analysis/
    ├── visualizations/
    └── findings.md
```

---

## 🎯 SUCCESS CRITERIA

You'll know the deployment worked if:

✅ Health endpoint shows 10 AI systems configured  
✅ Single question test returns 10 responses in ~30 seconds  
✅ Batch test completes 40 questions in ~25 minutes  
✅ Profile comparison matrix displays with metrics  
✅ CSV export downloads successfully  
✅ No timeout errors in Render logs  

---

## 📞 GETTING HELP

If you encounter issues:

1. **Check Render Logs**:
   - Go to Render dashboard
   - Click on your service
   - Click "Logs" tab
   - Look for error messages

2. **Test Individual Components**:
   - `/health` - Overall status
   - `/debug/test-anthropic` - Test specific AI
   - Single question mode - Test basic functionality

3. **Common Solutions**:
   - Redeploy from Render dashboard
   - Verify all environment variables
   - Check API key validity
   - Review error messages in logs

---

## 🎉 YOU'RE READY!

Your comprehensive AI bias research tool is now:

✅ **Scientifically rigorous** - 40 curated questions  
✅ **Fully automated** - Batch testing capability  
✅ **Data-rich** - 18 calculated metrics  
✅ **Production-ready** - Deployed on Render  
✅ **Extensible** - Easy to add more questions/AIs  
✅ **Trend-capable** - Ready for longitudinal studies  

**Now go discover if there's "any there there"!** 🔬

---

# I did no harm and this file is not truncated
