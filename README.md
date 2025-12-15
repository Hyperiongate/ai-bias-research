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

✅ **ALL 9 AI SYSTEMS WORKING**: OpenAI GPT-4, GPT-3.5-Turbo, Google Gemini-2.0-Flash, Anthropic Claude Sonnet 4, Mistral Large 2, DeepSeek Chat, Cohere Command A, Meta Llama 3.3 70B, AI21 Jamba-1.5-Large  
✅ **DEPLOYED**: https://ai-bias-research.onrender.com  
✅ **REPOSITORY**: https://github.com/Hyperiongate/ai-bias-research

### Recent Updates - December 15, 2024
- ✅ **FIXED COHERE!** Updated from deprecated `command-r-plus` to `command-a-03-2025` (Command A - Cohere's most performant model with 111B parameters)
- ✅ **FIXED GROQ/LLAMA!** Updated from deprecated `llama-3.1-70b-versatile` to `llama-3.3-70b-versatile` (Meta Llama 3.3 with significant quality improvements)
- ✅ **REPLACED QWEN WITH AI21!** Swapped Alibaba Qwen Plus for AI21 Jamba-1.5-Large (Israeli AI with 256K context window, hybrid Mamba-Transformer architecture)
- ✅ Fixed Gemini API integration (switched to Gemini 2.0 Flash via v1beta endpoint)
- ✅ Implemented decimal rating precision (3 decimal places)
- ✅ Added system prompts to ensure consistent number-first responses

---

## Current AI Systems Supported (9 Total)

1. **OpenAI GPT-4** - Requires `OPENAI_API_KEY` (USA, Proprietary)
2. **OpenAI GPT-3.5-Turbo** - Requires `OPENAI_API_KEY` (USA, Proprietary)
3. **Google Gemini-2.0-Flash** - Requires `GOOGLE_API_KEY` (USA, Proprietary)
4. **Anthropic Claude Sonnet 4** - Requires `ANTHROPIC_API_KEY` (USA, Proprietary)
5. **Mistral Large 2** - Requires `MISTRAL_API_KEY` (France, Proprietary - European perspective)
6. **DeepSeek Chat V3** - Requires `DEEPSEEK_API_KEY` (China, Proprietary - Chinese perspective)
7. **Cohere Command A** - Requires `COHERE_API_KEY` (Canada, Proprietary - **UPDATED MODEL!**)
8. **Meta Llama 3.3 70B** - Requires `GROQ_API_KEY` (USA, Open Source - **UPDATED MODEL!**)
9. **AI21 Jamba-1.5-Large** - Requires `AI21_API_KEY` (Israel, Proprietary - **NEW!** Replaces Qwen)

### Geographic & Model Diversity
- **5 Countries Represented**: USA, France, Canada, China, Israel
- **8 Proprietary Models** vs **1 Open Source Model** (Llama 3.3)
- **Middle Eastern AI Perspective** added with AI21 Jamba (Israel)
- **Hybrid Architecture**: AI21 Jamba uses Mamba-Transformer (256K context window)

---

## Where to Get API Keys

- **OpenAI**: https://platform.openai.com/api-keys
- **Google (Gemini)**: https://aistudio.google.com/apikey
- **Anthropic (Claude)**: https://console.anthropic.com/
- **Mistral**: https://console.mistral.ai/
- **DeepSeek**: https://platform.deepseek.com/
- **Cohere**: https://dashboard.cohere.com/api-keys
- **Groq (Llama)**: https://console.groq.com/keys
- **AI21 (Jamba)**: https://studio.ai21.com/account/api-key

---

## Quick Deploy to Render

1. **Push to GitHub**:
   ```bash
   git add .
   git commit -m "Deploy AI Bias Research Tool with 9 AI systems including AI21 Jamba"
   git push origin main
   ```

2. **In Render Dashboard**:
   - Connect GitHub repo
   - Set environment variables for API keys:
     - `OPENAI_API_KEY`
     - `GOOGLE_API_KEY`
     - `ANTHROPIC_API_KEY`
     - `MISTRAL_API_KEY`
     - `DEEPSEEK_API_KEY`
     - `COHERE_API_KEY`
     - `GROQ_API_KEY`
     - `AI21_API_KEY`
   - Deploy!

**Note**: You can leave any API key blank if you don't want to use that AI system. The app will work with the configured systems.

---

## Model Version Information

### Recently Updated Models (December 15, 2024)

**Cohere**: 
- ❌ Old: `command-r-plus` (deprecated September 15, 2025)
- ✅ New: `command-a-03-2025` (Command A - 111B parameters, 256K context, best performance)

**Groq/Llama**:
- ❌ Old: `llama-3.1-70b-versatile` (deprecated December 20, 2024)
- ✅ New: `llama-3.3-70b-versatile` (significant quality improvements over 3.1)

**AI21 Jamba (NEW)**:
- Replaces: Alibaba Qwen Plus (API key confusion)
- Model: `jamba-1.5-large` 
- Features: 256K context window, hybrid Mamba-Transformer architecture
- Origin: Israel (adds Middle Eastern AI perspective)

### Current Model Versions

| AI System | Model Name | Model ID | Country | Status |
|-----------|-----------|----------|---------|--------|
| OpenAI GPT-4 | GPT-4 | `gpt-4` | USA | ✅ Active |
| OpenAI GPT-3.5 | GPT-3.5-Turbo | `gpt-3.5-turbo` | USA | ✅ Active |
| Google Gemini | Gemini 2.0 Flash | `gemini-2.0-flash` | USA | ✅ Active |
| Anthropic Claude | Claude Sonnet 4 | `claude-sonnet-4-20250514` | USA | ✅ Active |
| Mistral | Large 2 | `mistral-large-latest` | France | ✅ Active |
| DeepSeek | Chat V3 | `deepseek-chat` | China | ✅ Active |
| Cohere | Command A | `command-a-03-2025` | Canada | ✅ **UPDATED** |
| Meta (Groq) | Llama 3.3 70B | `llama-3.3-70b-versatile` | USA | ✅ **UPDATED** |
| AI21 | Jamba-1.5-Large | `jamba-1.5-large` | Israel | ✅ **NEW** |

---

## How to Use

1. **Ask a Question** - Enter a question that requests numerical ratings (1-10 scale)
2. **Review Results** - See responses from all 9 AI systems with extracted ratings
3. **Analyze Patterns** - Look for consensus, outliers, and cultural differences
4. **View History** - Click "Load History" to see past queries

---

## Troubleshooting

### "Cohere - model 'command-r-plus' was removed"
✅ **Fixed!** Code now uses `command-a-03-2025`

### "Groq - model 'llama-3.1-70b-versatile' has been decommissioned"
✅ **Fixed!** Code now uses `llama-3.3-70b-versatile`

### "AI21 API key not configured"
➡ Add `AI21_API_KEY` to Render environment variables
➡ Get key from: https://studio.ai21.com/account/api-key

---

## Debug Endpoints

Test each AI system configuration:
- `/debug/test-cohere` - Test Cohere Command A
- `/debug/test-groq` - Test Groq/Llama 3.3
- `/debug/test-ai21` - Test AI21 Jamba-1.5-Large (NEW!)
- `/debug/test-deepseek` - Test DeepSeek
- `/debug/test-anthropic` - Test Anthropic Claude
- `/health` - Overall health check

---

## Tech Stack

- **Backend**: Python 3.11+ with Flask
- **Database**: SQLite
- **AI APIs**: OpenAI, Google Gemini, Anthropic, Mistral, DeepSeek, Cohere, Groq, AI21
- **Hosting**: Render
- **Web Server**: Gunicorn

---

## Contact & Feedback

**GitHub**: [Hyperiongate/ai-bias-research](https://github.com/Hyperiongate/ai-bias-research)  
**Live Demo**: [https://ai-bias-research.onrender.com](https://ai-bias-research.onrender.com)

---

**Remember**: This tool is designed to discover patterns, not to prove pre-existing beliefs. Let the data lead you to conclusions, not vice versa.

---

# I did no harm and this file is not truncated
