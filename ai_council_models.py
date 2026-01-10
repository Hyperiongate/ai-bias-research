"""
TruthLens AI Council - Database Models
File: ai_council_models.py
Date: January 9, 2026
Version: 1.0.0

PURPOSE:
Store questions asked to multiple AI services and their responses.
Users can ask any question and get perspectives from 7 different AIs:
- OpenAI GPT-4
- Anthropic Claude
- Mistral AI
- DeepSeek
- Cohere
- Groq
- xAI Grok

MODELS:
- AIQuery: User's question with metadata
- AIResponse: Individual AI service response
- AIConsensus: Consensus summary across all AIs

FEATURES:
- Store questions and responses
- Track which AIs responded
- Generate consensus summaries
- Extract claims from AI responses
- Search query history

INTEGRATION:
- Auto-extracts claims from AI responses
- Links to Claim Tracker database
- Saves claims as source_type: 'ai_consensus'

DO NO HARM: This is a NEW system - doesn't touch existing tables.

Last modified: January 9, 2026 - v1.0.0 Initial Release
I did no harm and this file is not truncated.
"""

from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

# Will be initialized with shared db instance from app.py
db = None

# Global model references
AIQuery = None
AIResponse = None
AIConsensus = None


def init_ai_council_db(shared_db):
    """
    Initialize AI Council models with SHARED database instance from app.py
    
    Follows the same pattern as claim_tracker_models.py - models are
    defined after db is set so SQLAlchemy can see them when db.create_all() is called.
    
    Args:
        shared_db: The SQLAlchemy database instance from app.py
        
    Returns:
        The same database instance (for consistency)
    """
    global db, AIQuery, AIResponse, AIConsensus
    
    db = shared_db
    
    # NOW define the models with proper columns
    
    class AIQuery(db.Model):
        """
        User question submitted to AI Council
        """
        __tablename__ = 'ai_queries'
        
        id = db.Column(db.Integer, primary_key=True)
        
        # Question details
        question = db.Column(db.Text, nullable=False)
        question_category = db.Column(db.String(100))
        # Categories: General, Political, Health, Science, Economics, Technology, Philosophy
        
        # Metadata
        created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
        processing_time = db.Column(db.Float)  # Seconds to get all responses
        
        # Response tracking
        total_responses = db.Column(db.Integer, default=0)
        successful_responses = db.Column(db.Integer, default=0)
        failed_responses = db.Column(db.Integer, default=0)
        
        # Consensus data
        has_consensus = db.Column(db.Boolean, default=False)
        consensus_level = db.Column(db.String(50))  # high, medium, low, conflicting
        
        # Claims extracted from responses
        claims_extracted = db.Column(db.Integer, default=0)
        
        # Relationships
        responses = db.relationship('AIResponse', back_populates='query', lazy='dynamic',
                                   cascade='all, delete-orphan')
        consensus = db.relationship('AIConsensus', back_populates='query', 
                                   uselist=False, cascade='all, delete-orphan')
        
        # Indexes for performance
        __table_args__ = (
            db.Index('idx_query_created', 'created_at'),
            db.Index('idx_query_category', 'question_category'),
        )
        
        def to_dict(self):
            """Convert query to dictionary for API responses"""
            return {
                'id': self.id,
                'question': self.question,
                'category': self.question_category,
                'created_at': self.created_at.isoformat() if self.created_at else None,
                'processing_time': self.processing_time,
                'total_responses': self.total_responses,
                'successful_responses': self.successful_responses,
                'failed_responses': self.failed_responses,
                'has_consensus': self.has_consensus,
                'consensus_level': self.consensus_level,
                'claims_extracted': self.claims_extracted,
                'response_count': self.responses.count()
            }
    
    
    class AIResponse(db.Model):
        """
        Individual AI service response to a question
        """
        __tablename__ = 'ai_responses'
        
        id = db.Column(db.Integer, primary_key=True)
        query_id = db.Column(db.Integer, db.ForeignKey('ai_queries.id'), nullable=False)
        
        # AI service information
        ai_service = db.Column(db.String(50), nullable=False)
        # Services: openai, anthropic, mistral, deepseek, cohere, groq, xai
        
        ai_model = db.Column(db.String(100))
        # Model used: gpt-4, claude-sonnet-4, etc.
        
        # Response data
        response_text = db.Column(db.Text)
        response_length = db.Column(db.Integer)  # Character count
        
        # Status
        success = db.Column(db.Boolean, default=True, nullable=False)
        error_message = db.Column(db.Text)
        
        # Performance
        response_time = db.Column(db.Float)  # Seconds
        tokens_used = db.Column(db.Integer)  # If available
        
        # Metadata
        created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
        
        # Sentiment/tone of response
        tone = db.Column(db.String(50))  # confident, uncertain, balanced, etc.
        
        # Relationship
        query = db.relationship('AIQuery', back_populates='responses')
        
        # Indexes
        __table_args__ = (
            db.Index('idx_response_query_service', 'query_id', 'ai_service'),
            db.Index('idx_response_success', 'success'),
        )
        
        def to_dict(self):
            """Convert response to dictionary for API responses"""
            return {
                'id': self.id,
                'ai_service': self.ai_service,
                'ai_model': self.ai_model,
                'response_text': self.response_text,
                'response_length': self.response_length,
                'success': self.success,
                'error_message': self.error_message,
                'response_time': self.response_time,
                'tokens_used': self.tokens_used,
                'tone': self.tone,
                'created_at': self.created_at.isoformat() if self.created_at else None
            }
    
    
    class AIConsensus(db.Model):
        """
        Consensus summary across all AI responses
        """
        __tablename__ = 'ai_consensus'
        
        id = db.Column(db.Integer, primary_key=True)
        query_id = db.Column(db.Integer, db.ForeignKey('ai_queries.id'), nullable=False)
        
        # Consensus summary
        summary = db.Column(db.Text)
        # Human-readable summary of what AIs agree/disagree on
        
        # Agreement analysis
        agreement_areas = db.Column(db.Text)
        # JSON or text of what AIs agreed on
        
        disagreement_areas = db.Column(db.Text)
        # JSON or text of what AIs disagreed on
        
        # Consensus metrics
        consensus_score = db.Column(db.Integer)  # 0-100
        # How much the AIs agreed
        
        # Key points
        key_points = db.Column(db.Text)
        # JSON array of main points from all responses
        
        # Generated by
        generated_by = db.Column(db.String(50))
        # Which AI generated the consensus summary
        
        # Metadata
        created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
        
        # Relationship
        query = db.relationship('AIQuery', back_populates='consensus')
        
        def to_dict(self):
            """Convert consensus to dictionary for API responses"""
            return {
                'id': self.id,
                'summary': self.summary,
                'agreement_areas': self.agreement_areas,
                'disagreement_areas': self.disagreement_areas,
                'consensus_score': self.consensus_score,
                'key_points': self.key_points,
                'generated_by': self.generated_by,
                'created_at': self.created_at.isoformat() if self.created_at else None
            }
    
    # Store model references globally
    globals()['AIQuery'] = AIQuery
    globals()['AIResponse'] = AIResponse
    globals()['AIConsensus'] = AIConsensus
    
    return db


# Helper functions

def categorize_question(question: str) -> str:
    """
    Automatically categorize a question based on keywords
    
    Args:
        question: The question text
        
    Returns:
        Category string
    """
    question_lower = question.lower()
    
    # Political keywords
    political_keywords = ['government', 'election', 'policy', 'president', 'congress', 
                         'politics', 'vote', 'democrat', 'republican', 'law', 'regulation']
    
    # Health keywords
    health_keywords = ['health', 'medical', 'disease', 'vaccine', 'treatment', 
                      'doctor', 'hospital', 'medicine', 'covid', 'cancer']
    
    # Science keywords
    science_keywords = ['science', 'research', 'study', 'climate', 'environment',
                       'physics', 'chemistry', 'biology', 'experiment', 'theory']
    
    # Economics keywords
    economics_keywords = ['economy', 'economic', 'money', 'inflation', 'gdp',
                         'market', 'stock', 'investment', 'trade', 'tax']
    
    # Technology keywords
    tech_keywords = ['technology', 'computer', 'software', 'ai', 'artificial intelligence',
                    'internet', 'cyber', 'digital', 'app', 'tech']
    
    # Philosophy keywords
    philosophy_keywords = ['philosophy', 'ethics', 'moral', 'meaning', 'existence',
                          'consciousness', 'free will', 'purpose', 'should']
    
    # Count matches
    if any(keyword in question_lower for keyword in political_keywords):
        return 'Political'
    elif any(keyword in question_lower for keyword in health_keywords):
        return 'Health'
    elif any(keyword in question_lower for keyword in science_keywords):
        return 'Science'
    elif any(keyword in question_lower for keyword in economics_keywords):
        return 'Economics'
    elif any(keyword in question_lower for keyword in tech_keywords):
        return 'Technology'
    elif any(keyword in question_lower for keyword in philosophy_keywords):
        return 'Philosophy'
    else:
        return 'General'


# I did no harm and this file is not truncated
