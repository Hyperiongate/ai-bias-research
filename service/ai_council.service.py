"""
AI Council Service - Multi-AI Query & Consensus Generation
File: services/ai_council_service.py
Date: January 9, 2026
Version: 1.0.0

PURPOSE:
Query multiple AI services with the same question and generate consensus.

AI SERVICES (7 total):
1. OpenAI GPT-4
2. Anthropic Claude Sonnet 4
3. Mistral Large
4. DeepSeek Chat
5. Cohere Command
6. Groq Llama
7. xAI Grok

FEATURES:
- Parallel execution (all 7 AIs queried simultaneously)
- Timeout handling (20s per AI)
- Error recovery (continues if some AIs fail)
- Consensus generation using Claude
- Claim extraction from responses

Last modified: January 9, 2026 - v1.0.0 Initial Release
I did no harm and this file is not truncated.
"""

import os
import logging
import time
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

logger = logging.getLogger(__name__)


class AICouncilService:
    """
    Query multiple AI services and generate consensus
    """
    
    def __init__(self):
        """Initialize AI clients"""
        self.ai_clients = {}
        self._initialize_clients()
        logger.info(f"[AICouncil] Initialized with {len(self.ai_clients)} AI services")
    
    def _initialize_clients(self):
        """Initialize all available AI clients"""
        
        # 1. OpenAI GPT-4
        try:
            import openai
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                self.ai_clients['openai'] = {
                    'client': openai.OpenAI(api_key=api_key),
                    'model': 'gpt-4',
                    'name': 'OpenAI GPT-4'
                }
                logger.info("✓ OpenAI GPT-4 initialized")
        except Exception as e:
            logger.warning(f"OpenAI unavailable: {e}")
        
        # 2. Anthropic Claude
        try:
            import anthropic
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if api_key:
                self.ai_clients['anthropic'] = {
                    'client': anthropic.Anthropic(api_key=api_key),
                    'model': 'claude-sonnet-4-20250514',
                    'name': 'Anthropic Claude Sonnet 4'
                }
                logger.info("✓ Anthropic Claude initialized")
        except Exception as e:
            logger.warning(f"Anthropic unavailable: {e}")
        
        # 3. Mistral
        try:
            from mistralai.client import MistralClient
            api_key = os.getenv('MISTRAL_API_KEY')
            if api_key:
                self.ai_clients['mistral'] = {
                    'client': MistralClient(api_key=api_key),
                    'model': 'mistral-large-latest',
                    'name': 'Mistral Large'
                }
                logger.info("✓ Mistral initialized")
        except Exception as e:
            logger.warning(f"Mistral unavailable: {e}")
        
        # 4. DeepSeek
        try:
            import openai  # DeepSeek uses OpenAI SDK
            api_key = os.getenv('DEEPSEEK_API_KEY')
            if api_key:
                self.ai_clients['deepseek'] = {
                    'client': openai.OpenAI(
                        api_key=api_key,
                        base_url="https://api.deepseek.com"
                    ),
                    'model': 'deepseek-chat',
                    'name': 'DeepSeek Chat'
                }
                logger.info("✓ DeepSeek initialized")
        except Exception as e:
            logger.warning(f"DeepSeek unavailable: {e}")
        
        # 5. Cohere
        try:
            import cohere
            api_key = os.getenv('COHERE_API_KEY')
            if api_key:
                self.ai_clients['cohere'] = {
                    'client': cohere.Client(api_key=api_key),
                    'model': 'command',
                    'name': 'Cohere Command'
                }
                logger.info("✓ Cohere initialized")
        except Exception as e:
            logger.warning(f"Cohere unavailable: {e}")
        
        # 6. Groq
        try:
            from groq import Groq
            api_key = os.getenv('GROQ_API_KEY')
            if api_key:
                self.ai_clients['groq'] = {
                    'client': Groq(api_key=api_key),
                    'model': 'llama-3.1-70b-versatile',
                    'name': 'Groq Llama 3.1 70B'
                }
                logger.info("✓ Groq initialized")
        except Exception as e:
            logger.warning(f"Groq unavailable: {e}")
        
        # 7. xAI Grok
        try:
            import openai  # xAI uses OpenAI SDK
            api_key = os.getenv('XAI_API_KEY')
            if api_key:
                self.ai_clients['xai'] = {
                    'client': openai.OpenAI(
                        api_key=api_key,
                        base_url="https://api.x.ai/v1"
                    ),
                    'model': 'grok-beta',
                    'name': 'xAI Grok'
                }
                logger.info("✓ xAI Grok initialized")
        except Exception as e:
            logger.warning(f"xAI unavailable: {e}")
    
    def query_all(self, question: str) -> Dict[str, Any]:
        """
        Query all AI services with the same question
        
        Args:
            question: User's question
            
        Returns:
            Dictionary with responses from all AIs + consensus
        """
        start_time = time.time()
        
        logger.info("=" * 80)
        logger.info(f"[AICouncil] Querying {len(self.ai_clients)} AI services")
        logger.info(f"[AICouncil] Question: {question[:100]}...")
        
        responses = []
        
        # Query all AIs in parallel
        with ThreadPoolExecutor(max_workers=7) as executor:
            futures = {}
            
            for service_name, client_info in self.ai_clients.items():
                future = executor.submit(
                    self._query_single_ai,
                    service_name,
                    client_info,
                    question
                )
                futures[future] = service_name
            
            # Collect results with timeout
            for future in as_completed(futures):
                service_name = futures[future]
                
                try:
                    result = future.result(timeout=20)  # 20s timeout per AI
                    if result:
                        responses.append(result)
                        logger.info(f"✓ {service_name}: response received")
                except TimeoutError:
                    logger.error(f"✗ {service_name}: TIMEOUT after 20s")
                    responses.append(self._get_error_response(service_name, "Timeout"))
                except Exception as e:
                    logger.error(f"✗ {service_name}: ERROR: {e}")
                    responses.append(self._get_error_response(service_name, str(e)))
        
        # Generate consensus using Claude (if available)
        consensus = self._generate_consensus(question, responses)
        
        processing_time = time.time() - start_time
        
        logger.info(f"[AICouncil] Complete: {len([r for r in responses if r['success']])}/{len(responses)} successful")
        logger.info(f"[AICouncil] Processing time: {processing_time:.2f}s")
        logger.info("=" * 80)
        
        return {
            'success': True,
            'question': question,
            'responses': responses,
            'consensus': consensus,
            'processing_time': processing_time,
            'total_responses': len(responses),
            'successful_responses': len([r for r in responses if r['success']]),
            'failed_responses': len([r for r in responses if not r['success']])
        }
    
    def _query_single_ai(self, service_name: str, client_info: Dict, question: str) -> Dict[str, Any]:
        """Query a single AI service"""
        start_time = time.time()
        
        try:
            client = client_info['client']
            model = client_info['model']
            
            # Query based on service type
            if service_name == 'openai':
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": question}],
                    max_tokens=1000,
                    temperature=0.7
                )
                response_text = response.choices[0].message.content
                tokens = response.usage.total_tokens if hasattr(response, 'usage') else None
            
            elif service_name == 'anthropic':
                response = client.messages.create(
                    model=model,
                    max_tokens=1000,
                    messages=[{"role": "user", "content": question}]
                )
                response_text = response.content[0].text
                tokens = response.usage.input_tokens + response.usage.output_tokens if hasattr(response, 'usage') else None
            
            elif service_name == 'mistral':
                response = client.chat(
                    model=model,
                    messages=[{"role": "user", "content": question}],
                    max_tokens=1000
                )
                response_text = response.choices[0].message.content
                tokens = response.usage.total_tokens if hasattr(response, 'usage') else None
            
            elif service_name == 'deepseek':
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": question}],
                    max_tokens=1000,
                    temperature=0.7
                )
                response_text = response.choices[0].message.content
                tokens = response.usage.total_tokens if hasattr(response, 'usage') else None
            
            elif service_name == 'cohere':
                response = client.chat(
                    model=model,
                    message=question,
                    max_tokens=1000
                )
                response_text = response.text
                tokens = None
            
            elif service_name == 'groq':
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": question}],
                    max_tokens=1000,
                    temperature=0.7
                )
                response_text = response.choices[0].message.content
                tokens = response.usage.total_tokens if hasattr(response, 'usage') else None
            
            elif service_name == 'xai':
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": question}],
                    max_tokens=1000,
                    temperature=0.7
                )
                response_text = response.choices[0].message.content
                tokens = response.usage.total_tokens if hasattr(response, 'usage') else None
            
            else:
                return self._get_error_response(service_name, "Unknown service type")
            
            response_time = time.time() - start_time
            
            return {
                'service': service_name,
                'name': client_info['name'],
                'model': model,
                'response': response_text,
                'response_length': len(response_text),
                'response_time': response_time,
                'tokens_used': tokens,
                'success': True,
                'error': None
            }
            
        except Exception as e:
            logger.error(f"Error querying {service_name}: {e}")
            return self._get_error_response(service_name, str(e))
    
    def _get_error_response(self, service_name: str, error: str) -> Dict[str, Any]:
        """Generate error response"""
        return {
            'service': service_name,
            'name': self.ai_clients.get(service_name, {}).get('name', service_name),
            'model': self.ai_clients.get(service_name, {}).get('model', 'unknown'),
            'response': None,
            'response_length': 0,
            'response_time': 0,
            'tokens_used': None,
            'success': False,
            'error': error
        }
    
    def _generate_consensus(self, question: str, responses: List[Dict]) -> Dict[str, Any]:
        """
        Generate consensus summary using Claude
        
        Args:
            question: Original question
            responses: List of AI responses
            
        Returns:
            Consensus summary with agreements/disagreements
        """
        try:
            # Only use successful responses
            successful_responses = [r for r in responses if r['success']]
            
            if len(successful_responses) < 2:
                return {
                    'summary': 'Insufficient responses to generate consensus',
                    'agreement_areas': [],
                    'disagreement_areas': [],
                    'consensus_score': 0,
                    'generated_by': None
                }
            
            # Use Claude to analyze consensus (if available)
            if 'anthropic' not in self.ai_clients:
                return self._simple_consensus(successful_responses)
            
            # Build prompt for Claude
            responses_text = "\n\n".join([
                f"**{r['name']}**: {r['response']}"
                for r in successful_responses
            ])
            
            prompt = f"""Analyze these {len(successful_responses)} AI responses to the question: "{question}"

Responses:
{responses_text}

Provide a consensus analysis:
1. What do the AIs AGREE on? (2-3 key points)
2. Where do they DISAGREE? (1-2 areas of conflict)
3. Overall consensus score (0-100, where 100 = total agreement)
4. Brief 2-sentence summary

Format as JSON:
{{
  "summary": "2-sentence summary",
  "agreement_areas": ["point 1", "point 2"],
  "disagreement_areas": ["conflict 1"],
  "consensus_score": 75
}}"""

            client = self.ai_clients['anthropic']['client']
            response = client.messages.create(
                model='claude-sonnet-4-20250514',
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            import json
            consensus_text = response.content[0].text
            
            # Extract JSON from response
            if consensus_text.startswith('```'):
                lines = consensus_text.split('\n')
                consensus_text = '\n'.join(lines[1:-1])
            
            consensus_data = json.loads(consensus_text)
            consensus_data['generated_by'] = 'claude'
            
            return consensus_data
            
        except Exception as e:
            logger.error(f"Error generating consensus: {e}")
            return self._simple_consensus(successful_responses)
    
    def _simple_consensus(self, responses: List[Dict]) -> Dict[str, Any]:
        """Simple consensus when Claude unavailable"""
        return {
            'summary': f'{len(responses)} AI services provided responses. Review individual responses for details.',
            'agreement_areas': ['Multiple perspectives available'],
            'disagreement_areas': [],
            'consensus_score': 50,
            'generated_by': 'simple'
        }


# I did no harm and this file is not truncated
