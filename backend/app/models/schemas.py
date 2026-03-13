"""
PitchPal v2 - Pydantic Models
Structured output schemas for pitch evaluation
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


# =============================
# Deck Upload Models
# =============================

class DeckQuality(BaseModel):
    """Visual and narrative quality scores for an uploaded pitch deck."""
    startup_name: str = ""
    design_score: float = Field(default=0.0, ge=0, le=10)
    narrative_score: float = Field(default=0.0, ge=0, le=10)
    data_viz_score: float = Field(default=0.0, ge=0, le=10)
    overall_deck_score: float = Field(default=0.0, ge=0, le=10)
    design_feedback: str = ""
    narrative_feedback: str = ""
    data_viz_feedback: str = ""
    strengths: List[str] = []
    improvements: List[str] = []
    analyzed_slides: int = 0


class DeckUploadResponse(BaseModel):
    """Response from POST /upload-deck"""
    startup_name: str
    extracted_text: str
    slide_count: int
    file_format: str  # "pdf" | "pptx"
    deck_quality: DeckQuality



# =============================
# Evaluation Models
# =============================

class DimensionScore(BaseModel):
    """Individual evaluation dimension with score and reasoning"""
    name: str = Field(description="Dimension name")
    score: float = Field(description="Score from 0 to 10", ge=0, le=10)
    reasoning: str = Field(description="Detailed reasoning for this score")
    suggestions: List[str] = Field(description="Actionable improvement suggestions")
    # Trust signals
    sources: List[str] = Field(default=[], description="URLs that support this dimension's score")
    benchmark: Optional[str] = Field(default=None, description="Industry benchmark comparison")


class Contradiction(BaseModel):
    """A claim in the pitch that is contradicted by live research"""
    pitch_claim: str = Field(description="What the founder claimed")
    research_finding: str = Field(description="What the research actually found")
    source: str = Field(default="", description="URL or domain where the contradicting evidence was found")


class PitchEvaluation(BaseModel):
    """Complete structured pitch evaluation — works for both startup (5 dims) and investor (7 dims)"""
    startup_name: str
    overall_score: float = Field(ge=0, le=10)
    investment_recommendation: str  # Buy / Hold / Pass
    role: str = "startup"  # "startup" | "investor"

    # Generic dimensions array — 5 for startup, 7 for investor
    dimensions: List[DimensionScore]

    # Insights
    key_strengths: List[str]
    main_concerns: List[str]
    next_steps: List[str]

    # Trust signals
    contradictions: List[Contradiction] = Field(default=[], description="Pitch claims that were contradicted by live research")


# ── Dimension definitions per role ──────────────────────────

STARTUP_DIMENSIONS = [
    "problem_clarity",
    "market_opportunity",
    "business_model",
    "competitive_advantage",
    "team_strength",
]

STARTUP_DIMENSION_TITLES = {
    "problem_clarity": "Problem Clarity",
    "market_opportunity": "Market Opportunity",
    "business_model": "Business Model",
    "competitive_advantage": "Competitive Advantage",
    "team_strength": "Team Strength",
}

INVESTOR_DIMENSIONS = [
    "market_opportunity",
    "revenue_economics",
    "scalability",
    "competitive_moat",
    "team_execution",
    "risk_assessment",
    "exit_potential",
]

INVESTOR_DIMENSION_TITLES = {
    "market_opportunity": "Market Opportunity",
    "revenue_economics": "Revenue & Unit Economics",
    "scalability": "Scalability",
    "competitive_moat": "Competitive Moat",
    "team_execution": "Team & Execution",
    "risk_assessment": "Risk Assessment",
    "exit_potential": "Exit Potential",
}


# =============================
# Agent Step Models (for streaming)
# =============================

class AgentStep(BaseModel):
    """A single ReAct agent step for live streaming"""
    step_number: int
    step_type: str  # "thought" | "action" | "observation" | "final_answer"
    content: str
    tool_name: Optional[str] = None
    tool_input: Optional[str] = None


# =============================
# API Models
# =============================

class PitchInput(BaseModel):
    """Input model for pitch evaluation"""
    startup_name: str = Field(min_length=1, max_length=200)
    pitch_text: str = Field(min_length=50, max_length=5000)


class EvaluationResponse(BaseModel):
    """Full evaluation response including agent steps"""
    startup_name: str
    evaluation: PitchEvaluation
    agent_steps: List[AgentStep]
    processing_time: float
    timestamp: datetime
    llm_provider: str


class HealthResponse(BaseModel):
    """API health check response"""
    status: str
    version: str
    llm_provider: str
    timestamp: datetime


# =============================
# Sample Pitches
# =============================

SAMPLE_PITCHES = [
    {
        "name": "HealthAI",
        "pitch": (
            "HealthAI uses machine learning to analyze medical images and detect "
            "early signs of diseases. Our proprietary AI model achieved 94% accuracy "
            "in detecting pneumonia from chest X-rays, outperforming average radiologist "
            "readings. We target rural hospitals and clinics in underserved areas that "
            "lack access to radiology specialists. Our SaaS pricing model charges $0.50 "
            "per scan with volume discounts. We currently have pilot programs running in "
            "5 hospitals across 3 states with over 10,000 scans processed. Our founding "
            "team includes a Stanford ML PhD, a former hospital CTO, and a healthcare "
            "sales veteran with 15 years experience. Seeking $2M seed round to scale "
            "our infrastructure and expand to 50 hospitals within 18 months."
        ),
    },
    {
        "name": "EduVerse",
        "pitch": (
            "EduVerse creates immersive VR learning experiences for K-12 students. "
            "We have built 100+ interactive VR lessons covering science, history, and "
            "geography. Pilot schools using EduVerse saw a 25% improvement in test scores "
            "and 40% increase in student engagement. We charge schools $30 per student "
            "per year with a freemium tier for individual teachers. Currently partnered "
            "with 15 schools reaching 3,000 students. Our team of 8 includes former "
            "educators, Unity developers, and an ex-Google product manager. Revenue is "
            "$90K ARR growing 20% month-over-month. Seeking $1M to expand content "
            "library and build a sales team for district-level deals."
        ),
    },
    {
        "name": "GreenCoin",
        "pitch": (
            "GreenCoin is a blockchain-based rewards platform that incentivizes "
            "sustainable behaviors. Users earn tokens for verified actions like recycling, "
            "using public transport, and reducing energy consumption. Tokens are redeemable "
            "for discounts at 50 partner businesses including local restaurants, gyms, and "
            "retail stores. We have 10,000 monthly active users with 65% retention rate. "
            "Revenue comes from transaction fees (2%) and business partnership fees. Our "
            "team includes a former Coinbase engineer, a sustainability consultant, and a "
            "growth marketer from Uber. Currently generating $15K MRR. Raising $500K to "
            "expand to 3 new cities and launch a carbon offset marketplace."
        ),
    },
]
