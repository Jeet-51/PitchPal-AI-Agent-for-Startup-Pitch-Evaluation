# pitch_evaluator.py
"""
PitchPal - AI-powered startup pitch evaluator
A portfolio project demonstrating LangChain, tools, async workflows, and OpenAI integration

Updated:
- Removed LangChain Agents imports (AgentExecutor, create_react_agent) to avoid Streamlit Cloud import errors
- Keeps tool-based evaluation + structured Pydantic output
- Keeps chain-based evaluator using LCEL (no LLMChain/SequentialChain)
"""

import json
import asyncio
from typing import Dict, List, Any
from pydantic import BaseModel, Field

# LangChain Core
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import BaseTool
from langchain_core.runnables import RunnablePassthrough

# OpenAI (LangChain wrapper)
from langchain_openai import ChatOpenAI

# Load environment variables
from dotenv import load_dotenv
load_dotenv()


# =============================
# Pydantic Models
# =============================
class DimensionScore(BaseModel):
    """Individual evaluation dimension"""
    name: str = Field(description="Dimension name")
    score: float = Field(description="Score 0-10", ge=0, le=10)
    reasoning: str = Field(description="Reasoning for score")
    suggestions: List[str] = Field(description="Improvement suggestions")


class PitchEvaluation(BaseModel):
    """Complete pitch evaluation"""
    startup_name: str
    overall_score: float = Field(ge=0, le=10)
    investment_recommendation: str

    problem_clarity: DimensionScore
    market_opportunity: DimensionScore
    business_model: DimensionScore
    competitive_advantage: DimensionScore
    team_strength: DimensionScore

    key_strengths: List[str]
    main_concerns: List[str]
    next_steps: List[str]


# =============================
# Custom Tools (simulated)
# =============================
class MarketResearchTool(BaseTool):
    """Tool for market research and analysis"""
    name: str = "market_research"
    description: str = "Analyzes market size, trends, and competition for a startup"

    def _run(self, query: str) -> str:
        industries = {
            "fintech": "Fintech market ~$312B, ~25% CAGR. Trends: embedded finance and AI risk scoring.",
            "healthtech": "Healthtech market ~$350B, ~15% CAGR. Trends: telemedicine and AI diagnostics.",
            "edtech": "Edtech market ~$89B, ~20% CAGR. Trends: personalized learning and remote delivery.",
            "default": "Market appears fragmented with moderate growth and room for differentiated entrants."
        }
        q = query.lower()
        for k, v in industries.items():
            if k in q:
                return v
        return industries["default"]

    async def _arun(self, query: str) -> str:
        return self._run(query)


class CompetitorAnalysisTool(BaseTool):
    """Tool for competitor analysis"""
    name: str = "competitor_analysis"
    description: str = "Identifies and analyzes competitors for a startup"

    def _run(self, query: str) -> str:
        competitors_db = {
            "ai": ["OpenAI", "Anthropic", "Google AI"],
            "fintech": ["Stripe", "Square", "PayPal"],
            "healthtech": ["Teladoc", "Oscar Health", "23andMe"],
            "edtech": ["Coursera", "Udemy", "Khan Academy"]
        }

        q = query.lower()
        for category, competitors in competitors_db.items():
            if category in q:
                return f"Key competitors: {', '.join(competitors[:3])}. Competitive market with differentiation opportunities."
        return "Competitive landscape appears fragmented with opportunities for a differentiated entrant."

    async def _arun(self, query: str) -> str:
        return self._run(query)


class FinancialModelingTool(BaseTool):
    """Tool for financial analysis and projections"""
    name: str = "financial_modeling"
    description: str = "Analyzes business model and financial projections"

    def _run(self, query: str) -> str:
        revenue_models = {
            "saas": "SaaS model supports recurring revenue; typical healthy LTV/CAC is ~3:1.",
            "marketplace": "Marketplace benefits from network effects; take rates often range 5-25%.",
            "subscription": "Subscription offers predictable revenue; churn management is critical.",
            "freemium": "Freemium relies on strong conversion (2-5%) and efficient upsell paths."
        }

        q = query.lower()
        for model, analysis in revenue_models.items():
            if model in q:
                return analysis
        return "Business model appears plausible; validate assumptions (pricing, churn, CAC, LTV) with pilots."

    async def _arun(self, query: str) -> str:
        return self._run(query)


# =============================
# Main Evaluator (tool-based, no agents)
# =============================
class EvaluationAgent:
    """Main evaluator that coordinates pitch analysis using tools + LLM (no LangChain agents)."""

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        self.tools = [
            MarketResearchTool(),
            CompetitorAnalysisTool(),
            FinancialModelingTool()
        ]

    async def evaluate_pitch(self, pitch_text: str, startup_name: str) -> PitchEvaluation:
        """Evaluate a startup pitch using tools + LLM. Falls back to direct evaluation if parsing fails."""
        try:
            return await self._tool_based_evaluation(pitch_text, startup_name)
        except Exception as e:
            print(f"Tool-based evaluation failed: {e}")
            return await self._direct_evaluation(pitch_text, startup_name)

    async def _tool_based_evaluation(self, pitch_text: str, startup_name: str) -> PitchEvaluation:
        # Step 1: Extract key info
        extraction_query = f"""
Analyze this startup pitch and extract key information:
Startup: {startup_name}
Pitch: {pitch_text}

Extract and format as bullet points:
- Industry
- Business model
- Target market
- Revenue model
- Key metrics
"""
        key_info_resp = await self.llm.ainvoke(extraction_query)
        key_info = key_info_resp.content if hasattr(key_info_resp, "content") else str(key_info_resp)

        # Step 2: Use tools directly (simulated)
        market_research = self.tools[0]._run(pitch_text)
        competitor_analysis = self.tools[1]._run(startup_name)
        financial_analysis = self.tools[2]._run(pitch_text)

        # Step 3: Ask for JSON evaluation
        evaluation_prompt = f"""
Based on this analysis, evaluate the startup pitch with scores 0-10.

Startup: {startup_name}
Pitch: {pitch_text}

Key Information:
{key_info}

Market Research:
{market_research}

Competitor Analysis:
{competitor_analysis}

Financial Analysis:
{financial_analysis}

Return STRICT JSON only in this schema:
{{
  "problem_clarity": {{"score": 0-10, "reasoning": "text", "suggestions": ["s1","s2"]}},
  "market_opportunity": {{"score": 0-10, "reasoning": "text", "suggestions": ["s1","s2"]}},
  "business_model": {{"score": 0-10, "reasoning": "text", "suggestions": ["s1","s2"]}},
  "competitive_advantage": {{"score": 0-10, "reasoning": "text", "suggestions": ["s1","s2"]}},
  "team_strength": {{"score": 0-10, "reasoning": "text", "suggestions": ["s1","s2"]}},
  "overall_score": 0-10,
  "recommendation": "Strong Buy/Buy/Hold/Pass/Strong Pass",
  "strengths": ["a","b","c"],
  "concerns": ["a","b","c"],
  "next_steps": ["a","b","c"]
}}
"""
        final_resp = await self.llm.ainvoke(evaluation_prompt)
        final_text = final_resp.content if hasattr(final_resp, "content") else str(final_resp)

        return self._parse_evaluation_response(final_text, startup_name)

    async def _direct_evaluation(self, pitch_text: str, startup_name: str) -> PitchEvaluation:
        direct_prompt = f"""
You are an expert startup evaluator. Analyze this pitch and provide structured evaluation.
Startup: {startup_name}
Pitch: {pitch_text}

Return STRICT JSON only in this schema:
{{
  "problem_clarity": {{"score": 0-10, "reasoning": "text", "suggestions": ["s1","s2"]}},
  "market_opportunity": {{"score": 0-10, "reasoning": "text", "suggestions": ["s1","s2"]}},
  "business_model": {{"score": 0-10, "reasoning": "text", "suggestions": ["s1","s2"]}},
  "competitive_advantage": {{"score": 0-10, "reasoning": "text", "suggestions": ["s1","s2"]}},
  "team_strength": {{"score": 0-10, "reasoning": "text", "suggestions": ["s1","s2"]}},
  "overall_score": 0-10,
  "recommendation": "Buy/Hold/Pass",
  "strengths": ["a","b","c"],
  "concerns": ["a","b","c"],
  "next_steps": ["a","b","c"]
}}
"""
        resp = await self.llm.ainvoke(direct_prompt)
        text = resp.content if hasattr(resp, "content") else str(resp)
        return self._parse_evaluation_response(text, startup_name)

    def _parse_evaluation_response(self, response: str, startup_name: str) -> PitchEvaluation:
        # Extract JSON safely
        cleaned = response.strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return self._create_fallback_evaluation(startup_name)

        try:
            data = json.loads(cleaned[start:end + 1])
        except Exception:
            return self._create_fallback_evaluation(startup_name)

        data = self._ensure_complete_evaluation(data)

        def build_dim(key: str, title: str) -> DimensionScore:
            d = data.get(key, {})
            return DimensionScore(
                name=title,
                score=float(d.get("score", 5.0)),
                reasoning=d.get("reasoning", "Analysis in progress"),
                suggestions=d.get("suggestions", ["Further validation recommended"])
            )

        evaluation = PitchEvaluation(
            startup_name=startup_name,
            overall_score=float(data.get("overall_score", 5.0)),
            investment_recommendation=data.get("recommendation", "Hold"),

            problem_clarity=build_dim("problem_clarity", "Problem Clarity"),
            market_opportunity=build_dim("market_opportunity", "Market Opportunity"),
            business_model=build_dim("business_model", "Business Model"),
            competitive_advantage=build_dim("competitive_advantage", "Competitive Advantage"),
            team_strength=build_dim("team_strength", "Team Strength"),

            key_strengths=data.get("strengths", []),
            main_concerns=data.get("concerns", []),
            next_steps=data.get("next_steps", [])
        )
        return evaluation

    def _ensure_complete_evaluation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        default_dim = {"score": 5.0, "reasoning": "Analysis completed", "suggestions": ["Further analysis recommended"]}
        for dim in ["problem_clarity", "market_opportunity", "business_model", "competitive_advantage", "team_strength"]:
            if dim not in data or not isinstance(data.get(dim), dict):
                data[dim] = default_dim.copy()
            else:
                if "score" not in data[dim]:
                    data[dim]["score"] = 5.0
                if "reasoning" not in data[dim]:
                    data[dim]["reasoning"] = "Analysis completed"
                if "suggestions" not in data[dim]:
                    data[dim]["suggestions"] = ["Further analysis recommended"]

        if "overall_score" not in data:
            scores = [float(data[d]["score"]) for d in ["problem_clarity", "market_opportunity", "business_model", "competitive_advantage", "team_strength"]]
            data["overall_score"] = sum(scores) / len(scores)

        if "recommendation" not in data:
            s = float(data["overall_score"])
            data["recommendation"] = "Buy" if s >= 7 else "Hold" if s >= 5 else "Pass"

        data.setdefault("strengths", ["Clear value proposition", "Market need", "Early traction"])
        data.setdefault("concerns", ["Competition", "Go-to-market risk", "Execution"])
        data.setdefault("next_steps", ["Validate assumptions", "Strengthen moat", "Pilot with customers"])
        return data

    def _create_fallback_evaluation(self, startup_name: str) -> PitchEvaluation:
        return PitchEvaluation(
            startup_name=startup_name,
            overall_score=5.0,
            investment_recommendation="Hold - Manual Review Needed",

            problem_clarity=DimensionScore(
                name="Problem Clarity",
                score=5.0,
                reasoning="Evaluation processing error - manual review needed",
                suggestions=["Clarify the problem statement and target customer"]
            ),
            market_opportunity=DimensionScore(
                name="Market Opportunity",
                score=5.0,
                reasoning="Evaluation processing error - manual review needed",
                suggestions=["Quantify TAM/SAM/SOM and buyer persona"]
            ),
            business_model=DimensionScore(
                name="Business Model",
                score=5.0,
                reasoning="Evaluation processing error - manual review needed",
                suggestions=["Validate pricing and unit economics"]
            ),
            competitive_advantage=DimensionScore(
                name="Competitive Advantage",
                score=5.0,
                reasoning="Evaluation processing error - manual review needed",
                suggestions=["Clarify differentiation and defensibility"]
            ),
            team_strength=DimensionScore(
                name="Team Strength",
                score=5.0,
                reasoning="Evaluation processing error - manual review needed",
                suggestions=["Highlight founder experience and execution plan"]
            ),

            key_strengths=["Manual evaluation needed"],
            main_concerns=["Processing error occurred"],
            next_steps=["Re-run evaluation after dependency fix", "Check logs", "Try again"]
        )


# =============================
# Simple Chain Evaluator (LCEL)
# =============================
class SimpleChainEvaluator:
    """LCEL-based evaluator (no legacy SequentialChain/LLMChain)."""

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        self._setup_chain()

    def _setup_chain(self):
        extraction_prompt = PromptTemplate(
            input_variables=["pitch"],
            template="""
Extract key information from this startup pitch:
{pitch}

Return as JSON-like text with keys:
industry, business_model, target_market, problem, solution, revenue_model, team
"""
        )

        evaluation_prompt = PromptTemplate(
            input_variables=["pitch", "extracted_info"],
            template="""
Based on this pitch and extracted info, provide a detailed evaluation:

Pitch: {pitch}

Extracted Info:
{extracted_info}

Include:
- strengths
- risks
- recommendation
"""
        )

        self.extract_chain = extraction_prompt | self.llm | StrOutputParser()
        self.eval_chain = evaluation_prompt | self.llm | StrOutputParser()

        self.full_chain = (
            {"pitch": RunnablePassthrough(), "extracted_info": self.extract_chain}
            | self.eval_chain
        )

    async def evaluate_pitch(self, pitch_text: str, startup_name: str) -> Dict[str, Any]:
        evaluation_text = await self.full_chain.ainvoke(pitch_text)
        extracted_info = await self.extract_chain.ainvoke({"pitch": pitch_text})

        return {
            "startup_name": startup_name,
            "extracted_info": extracted_info,
            "evaluation": evaluation_text,
            "method": "chain_based_lcel"
        }


# =============================
# Sample Pitches
# =============================
SAMPLE_PITCHES = [
    {
        "name": "HealthAI",
        "pitch": "HealthAI uses machine learning to analyze medical images and detect early signs of diseases. Our AI achieved 94% accuracy in detecting pneumonia from chest X-rays. We target rural hospitals lacking radiology specialists. SaaS pricing at $0.50 per scan with pilots in 5 hospitals. Seeking $2M to scale."
    },
    {
        "name": "EduVerse",
        "pitch": "EduVerse creates VR learning experiences for K-12 students with 100+ VR lessons. Pilot schools improved test scores by 25%. We charge $30/student/year and partner with 15 schools. Seeking $1M to scale content and sales."
    },
    {
        "name": "GreenCoin",
        "pitch": "GreenCoin is a blockchain platform rewarding sustainable behaviors like recycling and public transport. Users earn tokens redeemable for discounts at partner businesses. 10,000 active users, 50 partner businesses. Revenue from transaction fees and partnerships. Raising $500K to expand."
    }
]


# =============================
# Local Test (optional)
# =============================
async def test_evaluators():
    agent = EvaluationAgent()
    chain = SimpleChainEvaluator()

    sample = SAMPLE_PITCHES[0]

    print("Agent evaluation:")
    a = await agent.evaluate_pitch(sample["pitch"], sample["name"])
    print(a.overall_score, a.investment_recommendation)

    print("\nChain evaluation:")
    c = await chain.evaluate_pitch(sample["pitch"], sample["name"])
    print(c["extracted_info"][:200])


if __name__ == "__main__":
    asyncio.run(test_evaluators())
