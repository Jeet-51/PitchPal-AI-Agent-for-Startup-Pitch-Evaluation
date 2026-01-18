# pitch_evaluator.py
"""
PitchPal - AI-powered startup pitch evaluator
A portfolio project demonstrating LangChain, AI Agents, and OpenAI integration

NOTE:
- Uses modern LangChain (LCEL) APIs only
- No LLMChain / SequentialChain (avoids deprecation issues)
- Compatible with Streamlit Cloud
"""

import os
import json
import asyncio
from typing import Dict, List, Any
from datetime import datetime
from pydantic import BaseModel, Field

# LangChain Core
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import BaseTool
from langchain_core.runnables import RunnablePassthrough


# OpenAI
from langchain_openai import ChatOpenAI

# Load environment variables
from dotenv import load_dotenv
load_dotenv()


# =============================
# Pydantic Models
# =============================
class DimensionScore(BaseModel):
    name: str = Field(description="Dimension name")
    score: float = Field(description="Score 0-10", ge=0, le=10)
    reasoning: str
    suggestions: List[str]


class PitchEvaluation(BaseModel):
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
# Custom Tools
# =============================
class MarketResearchTool(BaseTool):
    name: str = "market_research"
    description: str = "Analyzes market size, trends, and competition"

    def _run(self, query: str) -> str:
        industries = {
            "fintech": "Fintech market ~$312B, ~25% CAGR. Trends: embedded finance, AI-driven risk.",
            "healthtech": "Healthtech market ~$350B, ~15% CAGR. Trends: telemedicine, AI diagnostics.",
            "edtech": "Edtech market ~$89B, ~20% CAGR. Trends: personalized and remote learning.",
            "default": "Emerging market with moderate growth and fragmentation."
        }
        q = query.lower()
        for k, v in industries.items():
            if k in q:
                return v
        return industries["default"]

    async def _arun(self, query: str) -> str:
        return self._run(query)


class CompetitorAnalysisTool(BaseTool):
    name: str = "competitor_analysis"
    description: str = "Identifies key competitors and competitive landscape"

    def _run(self, query: str) -> str:
        competitors = {
            "ai": ["OpenAI", "Anthropic", "Google AI"],
            "fintech": ["Stripe", "Square", "PayPal"],
            "healthtech": ["Teladoc", "Oscar Health", "23andMe"],
            "edtech": ["Coursera", "Udemy", "Khan Academy"],
        }
        q = query.lower()
        for k, v in competitors.items():
            if k in q:
                return f"Key competitors: {', '.join(v)}"
        return "Competitive landscape is fragmented with room for differentiation."

    async def _arun(self, query: str) -> str:
        return self._run(query)


class FinancialModelingTool(BaseTool):
    name: str = "financial_modeling"
    description: str = "Analyzes revenue model and unit economics"

    def _run(self, query: str) -> str:
        models = {
            "saas": "SaaS model with recurring revenue; healthy LTV/CAC ~3:1 benchmark.",
            "subscription": "Subscription model offers predictable revenue; churn is key risk.",
            "marketplace": "Marketplace benefits from network effects; take rates vary 5-25%.",
            "freemium": "Freemium requires strong conversion (2-5%) to be viable."
        }
        q = query.lower()
        for k, v in models.items():
            if k in q:
                return v
        return "Revenue model appears viable but assumptions need validation."

    async def _arun(self, query: str) -> str:
        return self._run(query)


# =============================
# ReAct Agent Evaluator
# =============================
class EvaluationAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        self.tools = [
            MarketResearchTool(),
            CompetitorAnalysisTool(),
            FinancialModelingTool()
        ]
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.agent = self._create_agent()

    def _create_agent(self) -> AgentExecutor:
        prompt_template = """
You are a startup investment analyst.
You have access to the following tools:

{tools}

Use this format:

Question: {input}
Thought: reasoning
Action: one of [{tool_names}]
Action Input: input
Observation: result
...
Thought: final reasoning
Final Answer: final answer
"""

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
            partial_variables={
                "tools": "\n".join([f"{t.name}: {t.description}" for t in self.tools]),
                "tool_names": ", ".join([t.name for t in self.tools])
            }
        )

        agent = create_react_agent(self.llm, self.tools, prompt)
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            max_iterations=3,
            handle_parsing_errors=True
        )

    async def evaluate_pitch(self, pitch_text: str, startup_name: str) -> PitchEvaluation:
        try:
            return await self._agent_evaluation(pitch_text, startup_name)
        except Exception:
            return await self._direct_evaluation(pitch_text, startup_name)

    async def _agent_evaluation(self, pitch_text: str, startup_name: str) -> PitchEvaluation:
        extraction_prompt = f"""
Analyze the following startup pitch and extract key details:

Startup: {startup_name}
Pitch: {pitch_text}

Return bullet points covering:
- Industry
- Target market
- Revenue model
- Key metrics
"""

        extracted = (await self.llm.ainvoke(extraction_prompt)).content

        market = self.tools[0]._run(pitch_text)
        competition = self.tools[1]._run(startup_name)
        finance = self.tools[2]._run(pitch_text)

        final_prompt = f"""
Evaluate the startup pitch using scores from 0-10.

Startup: {startup_name}
Pitch: {pitch_text}

Extracted Info:
{extracted}

Market:
{market}

Competition:
{competition}

Financials:
{finance}

Respond in JSON with:
problem_clarity, market_opportunity, business_model,
competitive_advantage, team_strength,
overall_score, recommendation, strengths, concerns, next_steps
"""

        response = (await self.llm.ainvoke(final_prompt)).content
        return self._parse_response(response, startup_name)

    async def _direct_evaluation(self, pitch_text: str, startup_name: str) -> PitchEvaluation:
        prompt = f"""
You are an experienced VC analyst.
Evaluate this pitch and return structured JSON.

Startup: {startup_name}
Pitch: {pitch_text}
"""
        response = (await self.llm.ainvoke(prompt)).content
        return self._parse_response(response, startup_name)

    def _parse_response(self, response: str, startup_name: str) -> PitchEvaluation:
        try:
            data = json.loads(response[response.find("{"):response.rfind("}")+1])
        except Exception:
            data = {}

        def dim(name):
            d = data.get(name, {})
            return DimensionScore(
                name=name.replace("_", " ").title(),
                score=float(d.get("score", 5.0)),
                reasoning=d.get("reasoning", "Analysis pending"),
                suggestions=d.get("suggestions", ["Further validation required"])
            )

        scores = [
            dim("problem_clarity").score,
            dim("market_opportunity").score,
            dim("business_model").score,
            dim("competitive_advantage").score,
            dim("team_strength").score
        ]

        return PitchEvaluation(
            startup_name=startup_name,
            overall_score=sum(scores) / len(scores),
            investment_recommendation=data.get("recommendation", "Hold"),
            problem_clarity=dim("problem_clarity"),
            market_opportunity=dim("market_opportunity"),
            business_model=dim("business_model"),
            competitive_advantage=dim("competitive_advantage"),
            team_strength=dim("team_strength"),
            key_strengths=data.get("strengths", []),
            main_concerns=data.get("concerns", []),
            next_steps=data.get("next_steps", [])
        )


# =============================
# LCEL Chain Evaluator
# =============================
class SimpleChainEvaluator:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        self._setup_chain()

    def _setup_chain(self):
        extraction_prompt = PromptTemplate(
            input_variables=["pitch"],
            template="""
Extract structured information from the pitch:
{pitch}
"""
        )

        evaluation_prompt = PromptTemplate(
            input_variables=["pitch", "extracted_info"],
            template="""
Evaluate the startup pitch:
Pitch: {pitch}
Extracted Info: {extracted_info}

Provide detailed analysis and recommendation.
"""
        )

        self.extract_chain = extraction_prompt | self.llm | StrOutputParser()
        self.eval_chain = evaluation_prompt | self.llm | StrOutputParser()

        self.full_chain = (
            {
                "pitch": RunnablePassthrough(),
                "extracted_info": self.extract_chain,
            }
            | self.eval_chain
        )

    async def evaluate_pitch(self, pitch_text: str, startup_name: str) -> Dict[str, Any]:
        evaluation = await self.full_chain.ainvoke(pitch_text)
        extracted = await self.extract_chain.ainvoke({"pitch": pitch_text})
        return {
            "startup_name": startup_name,
            "extracted_info": extracted,
            "evaluation": evaluation,
            "method": "chain_based_lcel"
        }


# =============================
# Sample Data
# =============================
SAMPLE_PITCHES = [
    {
        "name": "HealthAI",
        "pitch": "HealthAI uses machine learning to analyze medical images and detect early disease with 94% accuracy."
    },
    {
        "name": "EduVerse",
        "pitch": "EduVerse builds VR-based learning platforms for K-12 education with 25% test score improvement."
    },
    {
        "name": "GreenCoin",
        "pitch": "GreenCoin rewards sustainable behavior using blockchain-based incentives."
    }
]
