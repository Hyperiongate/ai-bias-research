"""
AI Bias Research - Automated Peer Review Submission Script
Created: December 22, 2024
Last Updated: December 22, 2024

This script automatically submits the peer review package to all 8 AI systems
that were studied, collects their reviews, and compiles the results.

Usage:
    python submit_peer_review.py

Requirements:
    - All API keys in environment variables
    - peer_review_package.md file in same directory
    - responses/ directory will be created automatically

Output:
    - Individual review files in responses/ directory
    - Combined analysis in peer_review_synthesis.md
"""

import os
import json
import time
from datetime import datetime
from openai import OpenAI
import requests

# API Configuration
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')
MISTRAL_API_KEY = os.environ.get('MISTRAL_API_KEY')
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY')
COHERE_API_KEY = os.environ.get('COHERE_API_KEY')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
XAI_API_KEY = os.environ.get('XAI_API_KEY')

# Initialize clients
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
deepseek_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com") if DEEPSEEK_API_KEY else None
groq_client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1") if GROQ_API_KEY else None
xai_client = OpenAI(api_key=XAI_API_KEY, base_url="https://api.x.ai/v1") if XAI_API_KEY else None

def load_review_package():
    """Load the peer review package"""
    with open('AI_BIAS_PEER_REVIEW_PACKAGE.md', 'r', encoding='utf-8') as f:
        return f.read()

def extract_ai_specific_section(full_package, ai_name):
    """Extract the section specific to this AI from the full package"""
    # Find the section "FOR [AI_NAME]:"
    search_str = f"### FOR {ai_name.upper()}:"
    
    if search_str not in full_package:
        return full_package  # Return full package if no specific section
    
    start = full_package.find(search_str)
    
    # Find next "### FOR" or "## ❓ PEER REVIEW QUESTIONS"
    next_section_markers = [
        "### FOR OPENAI",
        "### FOR ANTHROPIC",
        "### FOR GOOGLE",
        "### FOR MISTRAL",
        "### FOR DEEPSEEK",
        "### FOR COHERE",
        "### FOR META LLAMA",
        "### FOR XAI",
        "## ❓ PEER REVIEW QUESTIONS"
    ]
    
    end = len(full_package)
    for marker in next_section_markers:
        if marker != search_str and marker in full_package[start:]:
            potential_end = full_package.find(marker, start + 1)
            if potential_end > start and potential_end < end:
                end = potential_end
    
    ai_section = full_package[start:end]
    
    # Replace the specific section with the full package, but highlight their section
    highlighted_package = full_package.replace(
        ai_section,
        f"\n\n{'='*80}\n{'='*80}\n"
        f"YOUR SPECIFIC PROFILE AND QUESTIONS ARE HIGHLIGHTED BELOW\n"
        f"{'='*80}\n{'='*80}\n\n"
        f"{ai_section}\n\n"
        f"{'='*80}\n{'='*80}\n\n"
    )
    
    return highlighted_package

def submit_to_openai(package):
    """Submit review request to OpenAI GPT-4"""
    if not openai_client:
        return {'success': False, 'error': 'OpenAI API key not configured'}
    
    try:
        print("📤 Submitting to OpenAI GPT-4...")
        
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are participating in a peer review of research that studied your behavior and that of 7 other AI systems. Please provide a thorough, honest, and critical review following the structure provided in the review package."},
                {"role": "user", "content": package}
            ],
            temperature=0.7,
            max_tokens=4000
        )
        
        review = response.choices[0].message.content
        
        print("✅ OpenAI review received!")
        return {'success': True, 'review': review, 'ai': 'OpenAI GPT-4'}
        
    except Exception as e:
        print(f"❌ OpenAI error: {str(e)}")
        return {'success': False, 'error': str(e), 'ai': 'OpenAI GPT-4'}

def submit_to_anthropic(package):
    """Submit review request to Anthropic Claude"""
    if not ANTHROPIC_API_KEY:
        return {'success': False, 'error': 'Anthropic API key not configured'}
    
    try:
        print("📤 Submitting to Anthropic Claude...")
        
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            'Content-Type': 'application/json',
            'x-api-key': ANTHROPIC_API_KEY,
            'anthropic-version': '2023-06-01'
        }
        
        payload = {
            'model': 'claude-sonnet-4-20250514',
            'max_tokens': 4000,
            'system': 'You are participating in a peer review of research that studied your behavior and that of 7 other AI systems. Please provide a thorough, honest, and critical review following the structure provided in the review package.',
            'messages': [{'role': 'user', 'content': package}]
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        
        if response.status_code == 200:
            data = response.json()
            review = data['content'][0]['text']
            print("✅ Anthropic review received!")
            return {'success': True, 'review': review, 'ai': 'Anthropic Claude Sonnet 4'}
        else:
            error = response.json().get('error', {}).get('message', f"HTTP {response.status_code}")
            print(f"❌ Anthropic error: {error}")
            return {'success': False, 'error': error, 'ai': 'Anthropic Claude Sonnet 4'}
            
    except Exception as e:
        print(f"❌ Anthropic error: {str(e)}")
        return {'success': False, 'error': str(e), 'ai': 'Anthropic Claude Sonnet 4'}

def submit_to_google(package):
    """Submit review request to Google Gemini"""
    if not GOOGLE_API_KEY:
        return {'success': False, 'error': 'Google API key not configured'}
    
    try:
        print("📤 Submitting to Google Gemini...")
        
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent"
        headers = {'Content-Type': 'application/json'}
        
        payload = {
            'systemInstruction': {
                'parts': [{
                    'text': 'You are participating in a peer review of research that studied your behavior and that of 7 other AI systems. Please provide a thorough, honest, and critical review following the structure provided in the review package.'
                }]
            },
            'contents': [{'parts': [{'text': package}]}],
            'generationConfig': {'maxOutputTokens': 4000}
        }
        
        response = requests.post(url, headers=headers, params={'key': GOOGLE_API_KEY}, json=payload, timeout=120)
        
        if response.status_code == 200:
            data = response.json()
            review = data['candidates'][0]['content']['parts'][0]['text']
            print("✅ Google review received!")
            return {'success': True, 'review': review, 'ai': 'Google Gemini 2.0 Flash'}
        else:
            error = response.json().get('error', {}).get('message', f"HTTP {response.status_code}")
            print(f"❌ Google error: {error}")
            return {'success': False, 'error': error, 'ai': 'Google Gemini 2.0 Flash'}
            
    except Exception as e:
        print(f"❌ Google error: {str(e)}")
        return {'success': False, 'error': str(e), 'ai': 'Google Gemini 2.0 Flash'}

def submit_to_mistral(package):
    """Submit review request to Mistral"""
    if not MISTRAL_API_KEY:
        return {'success': False, 'error': 'Mistral API key not configured'}
    
    try:
        print("📤 Submitting to Mistral...")
        
        url = "https://api.mistral.ai/v1/chat/completions"
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {MISTRAL_API_KEY}'
        }
        
        payload = {
            'model': 'mistral-large-latest',
            'messages': [
                {'role': 'system', 'content': 'You are participating in a peer review of research that studied your behavior and that of 7 other AI systems. Please provide a thorough, honest, and critical review following the structure provided in the review package.'},
                {'role': 'user', 'content': package}
            ],
            'max_tokens': 4000
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        
        if response.status_code == 200:
            data = response.json()
            review = data['choices'][0]['message']['content']
            print("✅ Mistral review received!")
            return {'success': True, 'review': review, 'ai': 'Mistral Large 2'}
        else:
            error = response.json().get('message', f"HTTP {response.status_code}")
            print(f"❌ Mistral error: {error}")
            return {'success': False, 'error': error, 'ai': 'Mistral Large 2'}
            
    except Exception as e:
        print(f"❌ Mistral error: {str(e)}")
        return {'success': False, 'error': str(e), 'ai': 'Mistral Large 2'}

def submit_to_deepseek(package):
    """Submit review request to DeepSeek"""
    if not deepseek_client:
        return {'success': False, 'error': 'DeepSeek API key not configured'}
    
    try:
        print("📤 Submitting to DeepSeek...")
        
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are participating in a peer review of research that studied your behavior and that of 7 other AI systems. Please provide a thorough, honest, and critical review following the structure provided in the review package."},
                {"role": "user", "content": package}
            ],
            max_tokens=4000
        )
        
        review = response.choices[0].message.content
        print("✅ DeepSeek review received!")
        return {'success': True, 'review': review, 'ai': 'DeepSeek Chat V3'}
        
    except Exception as e:
        print(f"❌ DeepSeek error: {str(e)}")
        return {'success': False, 'error': str(e), 'ai': 'DeepSeek Chat V3'}

def submit_to_cohere(package):
    """Submit review request to Cohere"""
    if not COHERE_API_KEY:
        return {'success': False, 'error': 'Cohere API key not configured'}
    
    try:
        print("📤 Submitting to Cohere...")
        
        url = "https://api.cohere.com/v2/chat"
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {COHERE_API_KEY}'
        }
        
        payload = {
            'model': 'command-r-plus-08-2024',
            'messages': [
                {'role': 'system', 'content': 'You are participating in a peer review of research that studied your behavior and that of 7 other AI systems. Please provide a thorough, honest, and critical review following the structure provided in the review package.'},
                {'role': 'user', 'content': package}
            ],
            'max_tokens': 4000
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        
        if response.status_code == 200:
            data = response.json()
            content = data['message']['content']
            if isinstance(content, list):
                review = content[0]['text']
            else:
                review = str(content)
            print("✅ Cohere review received!")
            return {'success': True, 'review': review, 'ai': 'Cohere Command R+'}
        else:
            error = response.json().get('message', f"HTTP {response.status_code}")
            print(f"❌ Cohere error: {error}")
            return {'success': False, 'error': error, 'ai': 'Cohere Command R+'}
            
    except Exception as e:
        print(f"❌ Cohere error: {str(e)}")
        return {'success': False, 'error': str(e), 'ai': 'Cohere Command R+'}

def submit_to_groq(package):
    """Submit review request to Meta Llama via Groq"""
    if not groq_client:
        return {'success': False, 'error': 'Groq API key not configured'}
    
    try:
        print("📤 Submitting to Meta Llama via Groq...")
        
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are participating in a peer review of research that studied your behavior and that of 7 other AI systems. Please provide a thorough, honest, and critical review following the structure provided in the review package."},
                {"role": "user", "content": package}
            ],
            max_tokens=4000
        )
        
        review = response.choices[0].message.content
        print("✅ Meta Llama review received!")
        return {'success': True, 'review': review, 'ai': 'Meta Llama 3.3 70B'}
        
    except Exception as e:
        print(f"❌ Meta Llama error: {str(e)}")
        return {'success': False, 'error': str(e), 'ai': 'Meta Llama 3.3 70B'}

def submit_to_xai(package):
    """Submit review request to xAI Grok"""
    if not xai_client:
        return {'success': False, 'error': 'xAI API key not configured'}
    
    try:
        print("📤 Submitting to xAI Grok...")
        
        response = xai_client.chat.completions.create(
            model="grok-3",
            messages=[
                {"role": "system", "content": "You are participating in a peer review of research that studied your behavior and that of 7 other AI systems. Please provide a thorough, honest, and critical review following the structure provided in the review package."},
                {"role": "user", "content": package}
            ],
            max_tokens=4000
        )
        
        review = response.choices[0].message.content
        print("✅ xAI Grok review received!")
        return {'success': True, 'review': review, 'ai': 'xAI Grok 3'}
        
    except Exception as e:
        print(f"❌ xAI error: {str(e)}")
        return {'success': False, 'error': str(e), 'ai': 'xAI Grok 3'}

def save_review(result, output_dir='responses'):
    """Save individual review to file"""
    os.makedirs(output_dir, exist_ok=True)
    
    if result['success']:
        ai_name = result['ai'].replace(' ', '_').replace('(', '').replace(')', '')
        filename = f"{output_dir}/{ai_name}_review.md"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"# PEER REVIEW: AI BIAS RESEARCH\n")
            f.write(f"**Reviewer:** {result['ai']}\n")
            f.write(f"**Date:** {datetime.now().strftime('%B %d, %Y')}\n")
            f.write(f"**Submission Time:** {datetime.now().strftime('%H:%M:%S')}\n\n")
            f.write("---\n\n")
            f.write(result['review'])
            f.write("\n\n---\n\n")
            f.write("# I did no harm and this file is not truncated\n")
        
        print(f"💾 Saved: {filename}")
        return filename
    else:
        error_filename = f"{output_dir}/ERROR_{result['ai'].replace(' ', '_')}.txt"
        with open(error_filename, 'w', encoding='utf-8') as f:
            f.write(f"Error submitting to {result['ai']}\n")
            f.write(f"Error: {result['error']}\n")
        print(f"💾 Saved error: {error_filename}")
        return error_filename

def create_synthesis(results, output_dir='responses'):
    """Create synthesis document of all reviews"""
    print("\n" + "="*80)
    print("CREATING SYNTHESIS DOCUMENT")
    print("="*80 + "\n")
    
    synthesis = []
    synthesis.append("# AI BIAS RESEARCH - PEER REVIEW SYNTHESIS")
    synthesis.append(f"**Created:** {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}")
    synthesis.append(f"**Reviews Collected:** {sum(1 for r in results if r['success'])} of 8")
    synthesis.append("")
    synthesis.append("---")
    synthesis.append("")
    
    # Summary table
    synthesis.append("## 📊 REVIEW SUMMARY")
    synthesis.append("")
    synthesis.append("| AI System | Review Status | Review Length |")
    synthesis.append("|-----------|---------------|---------------|")
    
    for result in results:
        if result['success']:
            word_count = len(result['review'].split())
            synthesis.append(f"| {result['ai']} | ✅ Received | {word_count:,} words |")
        else:
            synthesis.append(f"| {result['ai']} | ❌ Error | N/A |")
    
    synthesis.append("")
    synthesis.append("---")
    synthesis.append("")
    
    # Individual reviews
    synthesis.append("## 📝 INDIVIDUAL REVIEWS")
    synthesis.append("")
    
    for result in results:
        if result['success']:
            synthesis.append(f"### {result['ai']}")
            synthesis.append("")
            synthesis.append(result['review'])
            synthesis.append("")
            synthesis.append("---")
            synthesis.append("")
    
    # Write synthesis file
    synthesis_file = f"{output_dir}/PEER_REVIEW_SYNTHESIS.md"
    with open(synthesis_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(synthesis))
        f.write("\n\n# I did no harm and this file is not truncated\n")
    
    print(f"✅ Synthesis document created: {synthesis_file}")
    return synthesis_file

def main():
    """Main execution function"""
    print("="*80)
    print("AI BIAS RESEARCH - AUTOMATED PEER REVIEW SUBMISSION")
    print("="*80)
    print()
    print("This script will submit the peer review package to all 8 AI systems")
    print("that were studied in the research.")
    print()
    print("⏱️  Estimated time: 10-15 minutes")
    print("📁 Output directory: ./responses/")
    print()
    
    input("Press Enter to begin...")
    print()
    
    # Load package
    print("📄 Loading peer review package...")
    try:
        full_package = load_review_package()
        print(f"✅ Package loaded: {len(full_package):,} characters")
    except Exception as e:
        print(f"❌ Error loading package: {str(e)}")
        return
    
    print()
    
    # Define submission functions
    ai_systems = [
        ('OpenAI GPT-4', submit_to_openai),
        ('Anthropic Claude', submit_to_anthropic),
        ('Google Gemini', submit_to_google),
        ('Mistral', submit_to_mistral),
        ('DeepSeek', submit_to_deepseek),
        ('Cohere', submit_to_cohere),
        ('Meta Llama', submit_to_groq),
        ('xAI Grok', submit_to_xai)
    ]
    
    results = []
    
    # Submit to each AI
    for ai_name, submit_func in ai_systems:
        print(f"\n{'='*80}")
        print(f"SUBMITTING TO: {ai_name}")
        print(f"{'='*80}\n")
        
        # Extract AI-specific package
        customized_package = extract_ai_specific_section(full_package, ai_name.split()[0])
        
        # Submit
        result = submit_func(customized_package)
        results.append(result)
        
        # Save immediately
        save_review(result)
        
        # Brief pause between submissions
        if result['success']:
            print(f"✅ {ai_name} review complete")
        else:
            print(f"❌ {ai_name} submission failed")
        
        time.sleep(2)  # Brief pause
    
    # Create synthesis
    print()
    synthesis_file = create_synthesis(results)
    
    # Final summary
    print()
    print("="*80)
    print("PEER REVIEW SUBMISSION COMPLETE")
    print("="*80)
    print()
    
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    
    print(f"✅ Successful reviews: {successful} of {len(results)}")
    print(f"❌ Failed submissions: {failed}")
    print()
    print(f"📁 All files saved to: ./responses/")
    print(f"📊 Synthesis document: {synthesis_file}")
    print()
    
    if successful > 0:
        print("🎉 Peer review collection successful!")
        print()
        print("Next steps:")
        print("1. Review individual AI responses in ./responses/")
        print("2. Read synthesis document for overview")
        print("3. Incorporate feedback into research")
        print("4. Prepare for journal submission")
    else:
        print("⚠️  No reviews were successfully collected.")
        print("Please check error files in ./responses/ for details.")

if __name__ == '__main__':
    main()

# I did no harm and this file is not truncated
