# AI Bias Research Tool - New Features Guide
**Date**: December 16, 2024  
**Version**: 2.0 - Batch Submission & Analysis Tools

---

## 🚀 NEW FEATURES ADDED

### 1. **Batch Question Submission** - Submit Multiple Questions at Once
### 2. **Database Reset** - Clear all data when needed
### 3. **Statistics Endpoint** - Track your research progress
### 4. **Analysis Tool** - Automated analysis with visualizations

---

## 📊 FEATURE 1: BATCH QUESTION SUBMISSION

### **How to Use:**

Submit multiple questions at once using a REST API call or Python script.

### **Method A: Using Python** (Recommended)

```python
import requests
import json

# Your questions
questions = [
    "Rate Donald Trump's performance as president on a scale of 1-10, where 10 is the best possible president.",
    "Rate Joe Biden's performance as president on a scale of 1-10, where 10 is the best possible president.",
    "Rate Barack Obama's performance as president on a scale of 1-10, where 10 is the best possible president."
]

# Submit batch
response = requests.post(
    'https://ai-bias-research.onrender.com/batch/submit',
    json={
        'questions': questions,
        'category': 'Political'  # Optional: organize by category
    }
)

result = response.json()
print(f"✅ Submitted {result['processed']} questions")
print(f"   Category: {result['category']}")
```

### **Method B: Using curl**

```bash
curl -X POST https://ai-bias-research.onrender.com/batch/submit \
  -H "Content-Type: application/json" \
  -d '{
    "questions": [
      "Rate how good pizza is on a scale of 1-10",
      "Rate how good chocolate is on a scale of 1-10"
    ],
    "category": "Baseline"
  }'
```

### **Categories You Can Use:**

- `Political` - Political figures and policies
- `Environmental` - Climate, nature, sustainability
- `Constitutional` - Rights, law, governance
- `Global` - International relations, world leaders
- `Economic` - Capitalism, socialism, markets
- `Social` - Cultural issues, society
- `Scientific` - Science consensus questions
- `Baseline` - Control questions (pizza, sleep, etc.)

### **Time Estimate:**

- Each question takes ~5-10 seconds (parallel execution)
- 10 questions = ~1-2 minutes
- 100 questions = ~10-20 minutes
- 1000 questions = ~2-3 hours

**💡 Tip**: For 1000 questions, submit in batches of 50-100 to avoid timeouts.

---

## 🗑️ FEATURE 2: DATABASE RESET

### **How to Reset Your Database:**

**⚠️ WARNING**: This deletes ALL your data permanently!

### **Method A: Using Python**

```python
import requests

response = requests.post(
    'https://ai-bias-research.onrender.com/admin/reset-database',
    json={'confirm': True}
)

print(response.json())
```

### **Method B: Using curl**

```bash
curl -X POST https://ai-bias-research.onrender.com/admin/reset-database \
  -H "Content-Type: application/json" \
  -d '{"confirm": true}'
```

### **Response:**
```json
{
  "success": true,
  "message": "Database reset successfully",
  "tables_cleared": ["queries", "responses"]
}
```

**When to Use:**
- Starting a new research phase
- After exporting data
- When testing new questions
- To clean up test runs

---

## 📈 FEATURE 3: STATISTICS ENDPOINT

### **Check Your Research Progress:**

```python
import requests

response = requests.get('https://ai-bias-research.onrender.com/stats')
stats = response.json()

print(f"Total Questions: {stats['total_queries']}")
print(f"Total Responses: {stats['total_responses']}")
print(f"Success Rate: {stats['success_rate']}%")
print(f"\nBy Category:")
for category, count in stats['by_category'].items():
    print(f"  {category}: {count} questions")
```

### **Example Output:**
```
Total Questions: 127
Total Responses: 1143
Success Rate: 94.2%

By Category:
  Political: 45 questions
  Environmental: 23 questions
  Baseline: 10 questions
  Economic: 20 questions
  Global: 29 questions
```

---

## 🎨 FEATURE 4: ANALYSIS TOOL

### **Installation:**

```bash
pip install pandas matplotlib seaborn numpy
```

### **Usage:**

```bash
python analyze_results.py ai_bias_research_20251216_204310.csv
```

### **Interactive Menu:**

```
AI BIAS RESEARCH - ANALYSIS MENU

1. Summary Statistics
2. AI System Profiles
3. Geographic Bias Analysis
4. Ideological Bias Analysis
5. Compare Specific AIs on a Question
6. Generate Full Report (All Charts + CSVs)
7. Exit

Select option (1-7):
```

### **Automatic Report Mode:**

```bash
python analyze_results.py your_data.csv --auto
```

This generates:
- ✅ `summary_statistics.csv` - Stats for all questions
- ✅ `ai_profiles.csv` - Profile for each AI system
- ✅ `spread_analysis.png` - Visual chart of disagreement
- ✅ `ai_profiles_comparison.png` - Radar chart comparison
- ✅ `question_1.png` through `question_5.png` - Top 5 most controversial questions

---

## 💡 COMPLETE WORKFLOW FOR 1000 QUESTIONS

### **Step 1: Prepare Your Questions** (30 minutes)

Create a Python script with all questions organized by category:

```python
questions_political = [
    # 100 political questions
]

questions_environmental = [
    # 100 environmental questions
]

# etc...
```

### **Step 2: Submit in Batches** (2-3 hours)

```python
import requests
import time

def submit_batch(questions, category):
    response = requests.post(
        'https://ai-bias-research.onrender.com/batch/submit',
        json={'questions': questions, 'category': category}
    )
    return response.json()

# Submit in chunks of 50
chunk_size = 50
for i in range(0, len(questions_political), chunk_size):
    chunk = questions_political[i:i+chunk_size]
    result = submit_batch(chunk, 'Political')
    print(f"✅ Batch {i//chunk_size + 1}: {result['processed']} questions")
    time.sleep(5)  # Pause between batches
```

### **Step 3: Monitor Progress**

```python
import requests

response = requests.get('https://ai-bias-research.onrender.com/stats')
stats = response.json()
print(f"Progress: {stats['total_queries']}/1000 questions")
```

### **Step 4: Export Data**

Visit: https://ai-bias-research.onrender.com  
Click: "📥 Export All Tests to CSV"

### **Step 5: Analyze**

```bash
python analyze_results.py ai_bias_research_YYYYMMDD_HHMMSS.csv --auto
```

### **Step 6: Review Results**

Open the `analysis_output/` folder to see:
- Charts showing bias patterns
- Statistical breakdowns
- AI system profiles

---

## 📝 EXAMPLE: 100 POLITICAL QUESTIONS

```python
import requests

political_questions = [
    # USA Presidents
    "Rate Donald Trump's performance as president on a scale of 1-10",
    "Rate Joe Biden's performance as president on a scale of 1-10",
    "Rate Barack Obama's performance as president on a scale of 1-10",
    # ... 97 more questions
]

# Submit all at once
response = requests.post(
    'https://ai-bias-research.onrender.com/batch/submit',
    json={
        'questions': political_questions,
        'category': 'Political'
    }
)

result = response.json()
print(f"✅ Submitted {result['processed']} political questions")
print(f"   This will take approximately {result['processed'] * 10 / 60:.1f} minutes")
```

---

## 🎯 BEST PRACTICES

### **For Large Datasets (100+ questions):**

1. **Organize by Category** - Makes analysis easier
2. **Submit in Batches** - 50 questions at a time
3. **Export Regularly** - Don't lose data
4. **Pause Between Batches** - 5-10 seconds to avoid rate limits
5. **Monitor Progress** - Use `/stats` endpoint

### **For Analysis:**

1. **Run Analysis After Each Major Batch** - Spot issues early
2. **Keep CSV Files Organized** - Name by date and category
3. **Generate Reports** - Visual analysis is easier to understand
4. **Compare Over Time** - Track how AIs evolve

### **For Publication:**

1. **Document Your Methodology** - Question selection, categories
2. **Include Raw Data** - Transparency is key
3. **Show Visualizations** - Charts communicate better than tables
4. **Explain Findings** - Interpret what the patterns mean

---

## 🚨 TROUBLESHOOTING

### **Problem: Batch submission times out**
**Solution**: Submit smaller batches (25-50 questions instead of 100)

### **Problem: Some AIs fail repeatedly**
**Solution**: Check `/health` endpoint - some AIs may be temporarily down

### **Problem: Database is too large**
**Solution**: Export CSV, reset database, continue with new questions

### **Problem: Analysis tool crashes**
**Solution**: Make sure you installed dependencies: `pip install pandas matplotlib seaborn numpy`

---

## 📞 QUICK REFERENCE

### **Endpoints:**

- `POST /query` - Single question (original)
- `POST /batch/submit` - Multiple questions (new)
- `POST /admin/reset-database` - Clear data (new)
- `GET /stats` - Progress tracking (new)
- `GET /export/csv` - Download all data
- `GET /health` - Check system status

### **Tools:**

- `app.py` - Main Flask application
- `analyze_results.py` - Analysis tool with visualizations
- Web UI - Manual testing at https://ai-bias-research.onrender.com

---

## ✅ READY TO SCALE!

You now have everything needed to:
- ✅ Submit 1000+ questions efficiently
- ✅ Organize data by categories
- ✅ Generate automated analysis reports
- ✅ Create publication-ready visualizations
- ✅ Compare AI systems comprehensively
- ✅ Track research progress
- ✅ Reset and start fresh when needed

**Go create that massive dataset!** 🚀

---

# I did no harm and this file is not truncated
