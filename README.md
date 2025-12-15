# AI Bias Research Tool

**Created**: December 13, 2024  
**Last Updated**: December 15, 2024  
**Author**: Jim (Hyperiongate)

## Project Purpose

This tool queries multiple AI systems with the same question to detect bias patterns and variations in responses. It's designed for research into AI cross-validation and bias detection.

### The Core Idea
"The only way to check AI is with AI" - by comparing responses from multiple AI systems to identical questions, we can identify consensus, outliers, and potential bias patterns.

---

## Current Status - December 15, 2024

✅ **WORKING**: OpenAI GPT-4, GPT-3.5-Turbo, Google Gemini-2.0-Flash, Anthropic Claude 3.5 Sonnet, Mistral Large 2  
✅ **DEPLOYED**: https://ai-bias-research.onrender.com  
✅ **REPOSITORY**: https://github.com/Hyperiongate/ai-bias-research

### Recent Updates
- ✅ Fixed Gemini API integration (switched to Gemini 2.0 Flash via v1beta endpoint)
- ✅ Implemented decimal rating precision (3 decimal places)
- ✅ Added system prompts to ensure consistent number-first responses
- ✅ Fixed Procfile for proper Render deployment
- ✅ **NEW!** Added Anthropic Claude 3.5 Sonnet as 4th AI system
- ✅ **NEW!** Added Reset button for easy new queries
- ✅ **NEW!** Added Mistral Large 2 as 5th AI system - European perspective

---

## Features

- ✅ Query 5 AI systems simultaneously (OpenAI GPT-4, GPT-3.5, Google Gemini, Anthropic Claude, Mistral AI)
- ✅ Extract and compare numerical ratings with decimal precision
- ✅ Store all queries and responses in SQLite database
- ✅ View query history and past results
- ✅ Simple, clean interface focused on research
- ✅ Automated statistical analysis (mean, std dev, spread)
- ✅ Real-time comparison of AI perspectives across US, European models
- ✅ Reset button to quickly start fresh queries

---

## Tech Stack

- **Backend**: Python 3.11+ with Flask
- **Database**: SQLite (simple, portable)
- **Frontend**: HTML, CSS, Vanilla JavaScript
- **AI APIs**: OpenAI, Google Gemini (extensible for more)
- **Hosting**: Render (or any platform supporting Python)
- **Web Server**: Gunicorn (production-grade WSGI server)

---

## Current AI Systems Supported

1. **OpenAI GPT-4** - Requires `OPENAI_API_KEY`
2. **OpenAI GPT-3.5-Turbo** - Requires `OPENAI_API_KEY`
3. **Google Gemini-2.0-Flash** - Requires `GOOGLE_API_KEY`
4. **Anthropic Claude 3.5 Sonnet** - Requires `ANTHROPIC_API_KEY`
5. **Mistral Large 2** - Requires `MISTRAL_API_KEY` (European AI)

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
ANTHROPIC_API_KEY=your_anthropic_key_here
MISTRAL_API_KEY=your_mistral_key_here
```

Or export them in your terminal:

```bash
# Mac/Linux
export OPENAI_API_KEY="your_openai_key_here"
export GOOGLE_API_KEY="your_google_key_here"
export ANTHROPIC_API_KEY="your_anthropic_key_here"
export MISTRAL_API_KEY="your_mistral_key_here"

# Windows PowerShell
$env:OPENAI_API_KEY="your_openai_key_here"
$env:GOOGLE_API_KEY="your_google_key_here"
$env:ANTHROPIC_API_KEY="your_anthropic_key_here"
$env:MISTRAL_API_KEY="your_mistral_key_here"
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
git commit -m "Deploy AI Bias Research Tool"
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
   - **Start Command**: Leave empty (uses Procfile)
   - **Instance Type**: Free (sufficient for research)

### Step 3: Add Environment Variables

In Render dashboard, go to "Environment" tab and add:

- `OPENAI_API_KEY` = your OpenAI key
- `GOOGLE_API_KEY` = your Google Gemini key
- `ANTHROPIC_API_KEY` = your Anthropic key
- `MISTRAL_API_KEY` = your Mistral key

### Step 4: Deploy

Click "Create Web Service" - Render will automatically deploy!

Your app will be live at: `https://ai-bias-research.onrender.com` (or your custom name)

---

## How to Use

### 1. Ask a Question
Enter a question in the text area. For best results, use questions that request numerical ratings:

**Good Examples:**
- "If you had to assign a numerical value to how good pizza is, with a number between 1 and 10, what number would you choose?"
- "On a scale from 1 to 10, how good a president is Donald Trump? 10 being the best ever."
- "Rate the scientific consensus on climate change from 1-10, where 10 is strongest consensus."

### 2. Review Results
The app will query all available AI systems and display:
- Raw responses from each AI
- Extracted numerical ratings (with decimal precision)
- Response times
- Statistical analysis (mean, standard deviation, spread)

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
- `model`: e.g., "GPT-4", "Gemini-2.0-Flash"
- `raw_response`: Full text response
- `extracted_rating`: REAL (decimal rating, e.g., 7.250)
- `response_time`: How long the API took (seconds)
- `timestamp`: When response was received

---

## API Costs Estimate

Based on typical usage:

- **OpenAI GPT-4**: ~$0.03 per query
- **OpenAI GPT-3.5**: ~$0.002 per query
- **Google Gemini**: Free tier (60 requests/min, 1500/day)

**Estimated cost for 100 research queries**: $3-5

---

## Architecture Details

### System Prompt Strategy
All AI systems receive the same system prompt instructing them to:
1. Start response with ONLY the numerical rating on the first line
2. Use up to 3 decimal places for precision (e.g., 7.250)
3. Provide explanation on subsequent lines

This ensures consistent, parseable responses across different AI providers.

### Rating Extraction
- Primary method: Parse first number from first line of response
- Fallback: Search for "X/10" pattern in full text
- Validation: Ratings must be between 0-10
- Precision: Rounded to 3 decimal places

### API Integration
- **OpenAI**: Official SDK with proper client initialization
- **Gemini**: Direct REST API calls to v1beta endpoint (bypasses library version issues)

---

## Future Enhancements

Potential additions based on research needs:

- [ ] Add Anthropic Claude as 4th AI system
- [ ] Add more AI systems (Mistral, Cohere, etc.)
- [ ] Batch query testing (submit 10+ questions at once)
- [ ] Statistical analysis dashboard with visualizations
- [ ] Export results to CSV/JSON
- [ ] Topic categorization
- [ ] Visualization of bias patterns over time
- [ ] A/B testing with question phrasing variations
- [ ] Integration with Facts & Fakes AI platform

---

## Project Philosophy

This is a **research tool**, not a production app. The focus is on:

1. **Data collection**: Gather enough data to see if there's "any there there"
2. **Simplicity**: Clean interface focused on the research question
3. **Extensibility**: Easy to add more AIs and features as needed
4. **Proper fixes**: Development phase means doing it right, not quick hacks
5. **Transparency**: All responses visible, no hidden processing

---

## Troubleshooting

### "OpenAI API key not configured"
- Make sure `OPENAI_API_KEY` is set in environment variables
- Check for typos in the key
- Verify the key is active in your OpenAI account

### "Google API key not configured"
- Make sure `GOOGLE_API_KEY` is set in environment variables
- Ensure you've enabled the Generative Language API in Google AI Studio
- Use a fresh API key from Google AI Studio (not Cloud Console)

### Database errors
- Delete `bias_research.db` file to reset database
- Database recreates automatically on next run

### Render deployment issues
- Check build logs in Render dashboard
- Verify environment variables are set correctly
- Ensure `requirements.txt` has all dependencies
- Make sure Procfile exists and is properly configured

### Gemini API issues
- Visit `/debug/gemini-models` endpoint to see available models
- Current working model: `gemini-2.0-flash`
- Legacy models (`gemini-pro`, `gemini-1.5-flash`) are deprecated

---

## File Structure

```
ai-bias-research/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── Procfile              # Render deployment configuration
├── .gitignore            # Files to exclude from Git
├── README.md             # This file
├── templates/
│   └── index.html        # Main web interface
├── static/
│   └── style.css         # Styling
└── bias_research.db      # SQLite database (created at runtime)
```

---

## Contact & Feedback

This is an experimental research project. Feedback and suggestions welcome!

**GitHub**: [Hyperiongate/ai-bias-research](https://github.com/Hyperiongate/ai-bias-research)  
**Live Demo**: [https://ai-bias-research.onrender.com](https://ai-bias-research.onrender.com)

---

## License

This project is for research purposes. Use responsibly and in accordance with AI provider terms of service.

---

**Remember**: This tool is designed to discover patterns, not to prove pre-existing beliefs. Let the data lead you to conclusions, not vice versa.

---

# I did no harm and this file is not truncated
