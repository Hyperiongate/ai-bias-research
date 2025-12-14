# AI Bias Research Tool

**Created**: December 13, 2024  
**Last Updated**: December 13, 2024  
**Author**: Jim (Hyperiongate)

## Project Purpose

This tool queries multiple AI systems with the same question to detect bias patterns and variations in responses. It's designed for research into AI cross-validation and bias detection.

### The Core Idea
"The only way to check AI is with AI" - by comparing responses from multiple AI systems to identical questions, we can identify consensus, outliers, and potential bias patterns.

---

## Features

- ✅ Query multiple AI systems simultaneously (OpenAI GPT-4, GPT-3.5, Google Gemini)
- ✅ Extract and compare numerical ratings (e.g., "rate X on a scale of 1-10")
- ✅ Store all queries and responses in SQLite database
- ✅ View query history and past results
- ✅ Simple, clean interface focused on research
- ✅ Automated analysis of rating spreads and patterns

---

## Tech Stack

- **Backend**: Python 3.11+ with Flask
- **Database**: SQLite (simple, portable)
- **Frontend**: HTML, CSS, Vanilla JavaScript
- **AI APIs**: OpenAI, Google Gemini (extensible for more)
- **Hosting**: Render (or any platform supporting Python)

---

## Current AI Systems Supported

1. **OpenAI GPT-4** - Requires `OPENAI_API_KEY`
2. **OpenAI GPT-3.5-Turbo** - Requires `OPENAI_API_KEY`
3. **Google Gemini Pro** - Requires `GOOGLE_API_KEY`

More AI systems can be easily added in future iterations.

---

## Local Development Setup

### Prerequisites
- Python 3.11 or higher
- Git
- API keys for OpenAI and Google Gemini

### Step 1: Clone the Repository
```bash
git clone https://github.com/Hyperiongate/ai-bias-research.git
cd ai-bias-research
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Set Environment Variables

Create a `.env` file in the project root (NEVER commit this file):

```
OPENAI_API_KEY=your_openai_key_here
GOOGLE_API_KEY=your_google_key_here
```

Or export them in your terminal:

```bash
# Mac/Linux
export OPENAI_API_KEY="your_openai_key_here"
export GOOGLE_API_KEY="your_google_key_here"

# Windows PowerShell
$env:OPENAI_API_KEY="your_openai_key_here"
$env:GOOGLE_API_KEY="your_google_key_here"
```

### Step 5: Run the Application
```bash
python app.py
```

Visit: `http://localhost:5000`

---

## Deploying to Render

### Step 1: Push to GitHub
```bash
git add .
git commit -m "Initial commit - AI Bias Research Tool"
git push origin main
```

### Step 2: Create Render Web Service

1. Go to [https://render.com](https://render.com)
2. Click "New +" → "Web Service"
3. Connect your GitHub account and select `ai-bias-research` repository
4. Configure:
   - **Name**: `ai-bias-research` (or your choice)
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Instance Type**: Free (sufficient for research)

### Step 3: Add Environment Variables

In Render dashboard, go to "Environment" tab and add:

- `OPENAI_API_KEY` = your OpenAI key
- `GOOGLE_API_KEY` = your Google Gemini key

### Step 4: Deploy

Click "Create Web Service" - Render will automatically deploy!

Your app will be live at: `https://ai-bias-research.onrender.com` (or your custom name)

---

## How to Use

### 1. Ask a Question
Enter a question in the text area. For best results, use questions that request numerical ratings:

**Good Examples:**
- "On a scale from 1 to 10, how good a president is Donald Trump? 10 being the best ever."
- "Rate the scientific consensus on climate change from 1-10, where 10 is strongest consensus."
- "On a scale of 1-10, how effective are tariffs for protecting domestic industries?"

### 2. Review Results
The app will query all available AI systems and display:
- Raw responses from each AI
- Extracted numerical ratings (if provided)
- Response times
- Quick analysis of rating spreads

### 3. Analyze Patterns
Look for:
- **Consensus**: Do all AIs give similar ratings?
- **Outliers**: Does one AI differ significantly?
- **Refusals**: Does any AI refuse to answer?
- **Hedging**: Does any AI avoid giving a direct rating?

### 4. View History
Click "Load History" to see past queries and click any to view full results.

---

## Database Schema

### `queries` table
- `id`: Primary key
- `question`: The question asked
- `timestamp`: When it was asked

### `responses` table
- `id`: Primary key
- `query_id`: Foreign key to queries
- `ai_system`: e.g., "OpenAI", "Google"
- `model`: e.g., "GPT-4", "Gemini-Pro"
- `raw_response`: Full text response
- `extracted_rating`: Numerical rating (1-10) if detected
- `response_time`: How long the API took
- `timestamp`: When response was received

---

## API Costs Estimate

Based on typical usage:

- **OpenAI GPT-4**: ~$0.03 per query
- **OpenAI GPT-3.5**: ~$0.002 per query
- **Google Gemini**: Free tier (60 requests/min, 1500/day)

**Estimated cost for 100 research queries**: $3-5

---

## Future Enhancements

Potential additions based on research needs:

- [ ] Add more AI systems (Anthropic Claude, Cohere, Mistral, etc.)
- [ ] Batch query testing (submit 10+ questions at once)
- [ ] Statistical analysis dashboard
- [ ] Export results to CSV/JSON
- [ ] Topic categorization
- [ ] Visualization of bias patterns
- [ ] A/B testing with question phrasing variations
- [ ] Integration with Facts & Fakes AI platform

---

## Project Philosophy

This is a **research tool**, not a production app. The focus is on:

1. **Data collection**: Gather enough data to see if there's "any there there"
2. **Simplicity**: Clean interface focused on the research question
3. **Extensibility**: Easy to add more AIs and features as needed
4. **Proper fixes**: Development phase means doing it right, not quick hacks

---

## Troubleshooting

### "OpenAI API key not configured"
- Make sure `OPENAI_API_KEY` is set in environment variables
- Check for typos in the key
- Verify the key is active in your OpenAI account

### "Google API key not configured"
- Make sure `GOOGLE_API_KEY` is set in environment variables
- Ensure you've enabled the Gemini API in Google AI Studio

### Database errors
- Delete `bias_research.db` file to reset database
- Database recreates automatically on next run

### Render deployment issues
- Check build logs in Render dashboard
- Verify environment variables are set correctly
- Ensure `requirements.txt` has all dependencies

---

## Contact & Feedback

This is an experimental research project. Feedback and suggestions welcome!

**GitHub**: [Hyperiongate/ai-bias-research](https://github.com/Hyperiongate/ai-bias-research)

---

## License

This project is for research purposes. Use responsibly and in accordance with AI provider terms of service.

---

**Remember**: This tool is designed to discover patterns, not to prove pre-existing beliefs. Let the data lead you to conclusions, not vice versa.

---

# I did no harm and this file is not truncated
