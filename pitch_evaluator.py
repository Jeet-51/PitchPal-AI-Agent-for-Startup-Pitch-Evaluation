"""
PitchPal - AI-powered startup pitch evaluator
A portfolio project demonstrating LangChain, AI Agents, and OpenAI integration
"""

import os
import json
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain.tools import BaseTool
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


# Pydantic Models for Structured Output
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
    
    # Core dimensions
    problem_clarity: DimensionScore
    market_opportunity: DimensionScore
    business_model: DimensionScore
    competitive_advantage: DimensionScore
    team_strength: DimensionScore
    
    # Summary insights
    key_strengths: List[str]
    main_concerns: List[str]
    next_steps: List[str]


# Custom Tools for AI Agents
class MarketResearchTool(BaseTool):
    """Tool for market research and analysis"""
    name: str = "market_research"
    description: str = "Analyzes market size, trends, and competition for a startup"
    
    def _run(self, query: str) -> str:
        """Simulate market research (in real app, would call external APIs)"""
        # Extract industry from query
        industries = {
            "fintech": "The global fintech market is valued at $312B and growing at 25% CAGR. Key trends include embedded finance and AI-powered solutions.",
            "healthtech": "Healthcare technology market is $350B with 15% CAGR. Telemedicine and AI diagnostics are major growth areas.",
            "edtech": "Education technology market is $89B growing at 20% CAGR. Personalized learning and remote education are key trends.",
            "default": "Market research indicates this is an emerging sector with significant growth potential."
        }
        
        query_lower = query.lower()
        for industry, data in industries.items():
            if industry in query_lower:
                return data
        return industries["default"]
    
    async def _arun(self, query: str) -> str:
        return self._run(query)


class CompetitorAnalysisTool(BaseTool):
    """Tool for competitor analysis"""
    name: str = "competitor_analysis"
    description: str = "Identifies and analyzes competitors for a startup"
    
    def _run(self, query: str) -> str:
        """Simulate competitor analysis"""
        # Simple competitor identification
        competitors_db = {
            "ai": ["OpenAI", "Anthropic", "Google AI"],
            "fintech": ["Stripe", "Square", "PayPal"],
            "healthtech": ["Teladoc", "23andMe", "Oscar Health"],
            "edtech": ["Coursera", "Udemy", "Khan Academy"]
        }
        
        query_lower = query.lower()
        for category, competitors in competitors_db.items():
            if category in query_lower:
                return f"Key competitors in this space include: {', '.join(competitors[:3])}. Market is competitive but has room for differentiated players."
        
        return "Competitive landscape appears fragmented with opportunities for new entrants."
    
    async def _arun(self, query: str) -> str:
        return self._run(query)


class FinancialModelingTool(BaseTool):
    """Tool for financial analysis and projections"""
    name: str = "financial_modeling"
    description: str = "Analyzes business model and financial projections"
    
    def _run(self, query: str) -> str:
        """Simulate financial analysis"""
        # Extract revenue model and provide analysis
        revenue_models = {
            "saas": "SaaS model shows strong recurring revenue potential. Industry average LTV/CAC ratio is 3:1.",
            "marketplace": "Marketplace model benefits from network effects. Take rates typically range from 3-30%.",
            "subscription": "Subscription model provides predictable revenue. Focus on churn reduction and customer acquisition.",
            "freemium": "Freemium model requires strong conversion rates (2-5%) from free to paid tiers."
        }
        
        query_lower = query.lower()
        for model, analysis in revenue_models.items():
            if model in query_lower:
                return analysis
        
        return "Financial model appears viable but requires validation of key assumptions and unit economics."
    
    async def _arun(self, query: str) -> str:
        return self._run(query)


# Specialized AI Agents
class EvaluationAgent:
    """Main evaluation agent that coordinates the pitch analysis"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0.3)
        self.tools = [
            MarketResearchTool(),
            CompetitorAnalysisTool(),
            FinancialModelingTool()
        ]
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        # Create ReAct agent
        self.agent = self._create_react_agent()
        
    def _create_react_agent(self):
        """Create a ReAct agent with tools"""
        prompt_template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["input", "agent_scratchpad"],
            partial_variables={
                "tools": "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools]),
                "tool_names": ", ".join([tool.name for tool in self.tools])
            }
        )
        
        agent = create_react_agent(self.llm, self.tools, prompt)
        return AgentExecutor(
            agent=agent, 
            tools=self.tools, 
            verbose=True, 
            memory=self.memory,
            handle_parsing_errors=True,  # Add this to handle parsing errors
            max_iterations=3,  # Limit iterations to prevent loops
            early_stopping_method="generate"  # Stop early if needed
        )
    
    async def evaluate_pitch(self, pitch_text: str, startup_name: str) -> PitchEvaluation:
        """Evaluate a startup pitch using AI agents and tools"""
        
        try:
            # Try the agent-based approach first
            return await self._agent_evaluation(pitch_text, startup_name)
        except Exception as e:
            print(f"Agent evaluation failed: {e}")
            # Fallback to direct LLM evaluation
            return await self._direct_evaluation(pitch_text, startup_name)
    
    async def _agent_evaluation(self, pitch_text: str, startup_name: str) -> PitchEvaluation:
        """Agent-based evaluation with tools"""
        
        # Step 1: Simple information extraction without agent
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
        
        key_info_response = await self.llm.ainvoke(extraction_query)
        key_info = key_info_response.content if hasattr(key_info_response, 'content') else str(key_info_response)
        
        # Step 2: Use tools directly (without agent for now)
        market_research = self.tools[0]._run(f"market research for {pitch_text[:100]}")
        competitor_analysis = self.tools[1]._run(f"competitor analysis for {startup_name}")
        financial_analysis = self.tools[2]._run(f"financial analysis for {pitch_text[:100]}")
        
        # Step 3: Generate final evaluation
        evaluation_prompt = f"""
        Based on this analysis, evaluate the startup pitch with scores 0-10:
        
        Startup: {startup_name}
        Pitch: {pitch_text}
        
        Key Information: {key_info}
        Market Research: {market_research}
        Competitor Analysis: {competitor_analysis}
        Financial Analysis: {financial_analysis}
        
        Provide JSON format response with:
        {{
            "problem_clarity": {{"score": 0-10, "reasoning": "explanation"}},
            "market_opportunity": {{"score": 0-10, "reasoning": "explanation"}},
            "business_model": {{"score": 0-10, "reasoning": "explanation"}},
            "competitive_advantage": {{"score": 0-10, "reasoning": "explanation"}},
            "team_strength": {{"score": 0-10, "reasoning": "explanation"}},
            "overall_score": 0-10,
            "recommendation": "Strong Buy/Buy/Hold/Pass/Strong Pass",
            "strengths": ["strength1", "strength2", "strength3"],
            "concerns": ["concern1", "concern2", "concern3"],
            "next_steps": ["step1", "step2", "step3"]
        }}
        """
        
        final_response = await self.llm.ainvoke(evaluation_prompt)
        final_evaluation = final_response.content if hasattr(final_response, 'content') else str(final_response)
        
        # Parse and structure the response
        return self._parse_evaluation_response(final_evaluation, startup_name)
    
    async def _direct_evaluation(self, pitch_text: str, startup_name: str) -> PitchEvaluation:
        """Direct LLM evaluation without agents (fallback)"""
        
        direct_prompt = f"""
        You are an expert startup evaluator. Analyze this pitch and provide structured evaluation:
        
        Startup: {startup_name}
        Pitch: {pitch_text}
        
        Score each dimension 0-10 and provide reasoning:
        
        1. Problem Clarity: How well-defined is the problem?
        2. Market Opportunity: How large and accessible is the market?
        3. Business Model: How viable is the revenue model?
        4. Competitive Advantage: How defensible is the position?
        5. Team Strength: How capable does the team appear?
        
        Provide response in this JSON format:
        {{
            "problem_clarity": {{"score": 8.0, "reasoning": "Clear problem definition..."}},
            "market_opportunity": {{"score": 7.5, "reasoning": "Large market with..."}},
            "business_model": {{"score": 7.0, "reasoning": "Viable SaaS model..."}},
            "competitive_advantage": {{"score": 6.5, "reasoning": "Some differentiation..."}},
            "team_strength": {{"score": 6.0, "reasoning": "Limited team info..."}},
            "overall_score": 7.0,
            "recommendation": "Buy",
            "strengths": ["Strong technology", "Clear market need", "Proven metrics"],
            "concerns": ["Competition risk", "Scaling challenges", "Regulatory hurdles"],
            "next_steps": ["Expand team", "Secure partnerships", "Scale technology"]
        }}
        """
        
        response = await self.llm.ainvoke(direct_prompt)
        evaluation_text = response.content if hasattr(response, 'content') else str(response)
        
        return self._parse_evaluation_response(evaluation_text, startup_name)
    
    def _parse_evaluation_response(self, response: str, startup_name: str) -> PitchEvaluation:
        """Parse agent response into structured evaluation"""
        try:
            # Clean the response and try to extract JSON
            cleaned_response = response.strip()
            
            # Try to find JSON in the response
            import re
            json_match = re.search(r'\{.*\}', cleaned_response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                eval_data = json.loads(json_str)
            else:
                # If no JSON found, create evaluation from text parsing
                eval_data = self._extract_scores_from_text(cleaned_response)
            
            # Ensure all required fields exist with defaults
            eval_data = self._ensure_complete_evaluation(eval_data)
            
            # Create structured evaluation
            evaluation = PitchEvaluation(
                startup_name=startup_name,
                overall_score=float(eval_data.get('overall_score', 5.0)),
                investment_recommendation=eval_data.get('recommendation', 'Hold'),
                
                problem_clarity=DimensionScore(
                    name="Problem Clarity",
                    score=float(eval_data.get('problem_clarity', {}).get('score', 5.0)),
                    reasoning=eval_data.get('problem_clarity', {}).get('reasoning', 'Analysis in progress'),
                    suggestions=eval_data.get('problem_clarity', {}).get('suggestions', ['Clarify problem definition'])
                ),
                
                market_opportunity=DimensionScore(
                    name="Market Opportunity",
                    score=float(eval_data.get('market_opportunity', {}).get('score', 5.0)),
                    reasoning=eval_data.get('market_opportunity', {}).get('reasoning', 'Analysis in progress'),
                    suggestions=eval_data.get('market_opportunity', {}).get('suggestions', ['Research market size'])
                ),
                
                business_model=DimensionScore(
                    name="Business Model",
                    score=float(eval_data.get('business_model', {}).get('score', 5.0)),
                    reasoning=eval_data.get('business_model', {}).get('reasoning', 'Analysis in progress'),
                    suggestions=eval_data.get('business_model', {}).get('suggestions', ['Validate revenue model'])
                ),
                
                competitive_advantage=DimensionScore(
                    name="Competitive Advantage",
                    score=float(eval_data.get('competitive_advantage', {}).get('score', 5.0)),
                    reasoning=eval_data.get('competitive_advantage', {}).get('reasoning', 'Analysis in progress'),
                    suggestions=eval_data.get('competitive_advantage', {}).get('suggestions', ['Strengthen differentiation'])
                ),
                
                team_strength=DimensionScore(
                    name="Team Strength",
                    score=float(eval_data.get('team_strength', {}).get('score', 5.0)),
                    reasoning=eval_data.get('team_strength', {}).get('reasoning', 'Analysis in progress'),
                    suggestions=eval_data.get('team_strength', {}).get('suggestions', ['Build stronger team'])
                ),
                
                key_strengths=eval_data.get('strengths', ['Strong market opportunity', 'Clear value proposition', 'Good initial traction']),
                main_concerns=eval_data.get('concerns', ['Competition risk', 'Market validation needed', 'Execution challenges']),
                next_steps=eval_data.get('next_steps', ['Build MVP', 'Validate market', 'Expand team'])
            )
            
            return evaluation
            
        except Exception as e:
            print(f"Error parsing evaluation: {e}")
            return self._create_fallback_evaluation(startup_name)
    
    def _extract_scores_from_text(self, text: str) -> Dict[str, Any]:
        """Extract scores from text when JSON parsing fails"""
        import re
        
        eval_data = {}
        
        # Look for scores in text
        score_patterns = [
            r'problem.{0,20}clarity.{0,20}(\d+(?:\.\d+)?)',
            r'market.{0,20}opportunity.{0,20}(\d+(?:\.\d+)?)',
            r'business.{0,20}model.{0,20}(\d+(?:\.\d+)?)',
            r'competitive.{0,20}advantage.{0,20}(\d+(?:\.\d+)?)',
            r'team.{0,20}strength.{0,20}(\d+(?:\.\d+)?)'
        ]
        
        dimension_names = ['problem_clarity', 'market_opportunity', 'business_model', 'competitive_advantage', 'team_strength']
        
        for i, pattern in enumerate(score_patterns):
            matches = re.findall(pattern, text.lower())
            if matches:
                score = float(matches[0])
                eval_data[dimension_names[i]] = {
                    'score': min(10.0, max(0.0, score)),
                    'reasoning': f'Extracted from evaluation text'
                }
        
        # Calculate overall score
        scores = [eval_data.get(dim, {}).get('score', 5.0) for dim in dimension_names]
        eval_data['overall_score'] = sum(scores) / len(scores)
        
        # Set recommendation based on score
        if eval_data['overall_score'] >= 8:
            eval_data['recommendation'] = 'Strong Buy'
        elif eval_data['overall_score'] >= 7:
            eval_data['recommendation'] = 'Buy'
        elif eval_data['overall_score'] >= 5:
            eval_data['recommendation'] = 'Hold'
        else:
            eval_data['recommendation'] = 'Pass'
        
        return eval_data
    
    def _ensure_complete_evaluation(self, eval_data: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure evaluation data has all required fields"""
        
        # Default dimension structure
        default_dimension = {'score': 5.0, 'reasoning': 'Analysis completed', 'suggestions': ['Further analysis recommended']}
        
        # Ensure all dimensions exist
        for dim in ['problem_clarity', 'market_opportunity', 'business_model', 'competitive_advantage', 'team_strength']:
            if dim not in eval_data:
                eval_data[dim] = default_dimension.copy()
            elif not isinstance(eval_data[dim], dict):
                eval_data[dim] = default_dimension.copy()
            else:
                # Ensure score exists and is valid
                if 'score' not in eval_data[dim]:
                    eval_data[dim]['score'] = 5.0
                eval_data[dim]['score'] = float(eval_data[dim]['score'])
                
                # Ensure reasoning exists
                if 'reasoning' not in eval_data[dim]:
                    eval_data[dim]['reasoning'] = 'Analysis completed'
                
                # Ensure suggestions exist
                if 'suggestions' not in eval_data[dim]:
                    eval_data[dim]['suggestions'] = ['Further analysis recommended']
        
        # Calculate overall score if missing
        if 'overall_score' not in eval_data:
            scores = [eval_data[dim]['score'] for dim in ['problem_clarity', 'market_opportunity', 'business_model', 'competitive_advantage', 'team_strength']]
            eval_data['overall_score'] = sum(scores) / len(scores)
        
        # Ensure other fields exist
        if 'recommendation' not in eval_data:
            score = eval_data['overall_score']
            if score >= 8:
                eval_data['recommendation'] = 'Strong Buy'
            elif score >= 7:
                eval_data['recommendation'] = 'Buy'
            elif score >= 5:
                eval_data['recommendation'] = 'Hold'
            else:
                eval_data['recommendation'] = 'Pass'
        
        if 'strengths' not in eval_data:
            eval_data['strengths'] = ['Strong market opportunity', 'Clear value proposition', 'Good initial metrics']
        
        if 'concerns' not in eval_data:
            eval_data['concerns'] = ['Competition risk', 'Market validation', 'Scaling challenges']
        
        if 'next_steps' not in eval_data:
            eval_data['next_steps'] = ['Validate market fit', 'Expand team', 'Secure funding']
        
        return eval_data
    
    def _fallback_parse(self, response: str) -> Dict[str, Any]:
        """Fallback parsing when JSON extraction fails"""
        # Simple keyword-based parsing
        scores = {}
        lines = response.split('\n')
        
        for line in lines:
            if 'problem' in line.lower() and any(str(i) in line for i in range(11)):
                scores['problem_clarity'] = float(re.search(r'(\d+)', line).group())
            elif 'market' in line.lower() and any(str(i) in line for i in range(11)):
                scores['market_opportunity'] = float(re.search(r'(\d+)', line).group())
            elif 'business' in line.lower() and any(str(i) in line for i in range(11)):
                scores['business_model'] = float(re.search(r'(\d+)', line).group())
        
        # Calculate overall score
        dimension_scores = [scores.get(dim, 5.0) for dim in ['problem_clarity', 'market_opportunity', 'business_model']]
        scores['overall_score'] = sum(dimension_scores) / len(dimension_scores)
        
        return scores
    
    def _create_fallback_evaluation(self, startup_name: str) -> PitchEvaluation:
        """Create fallback evaluation when parsing fails"""
        return PitchEvaluation(
            startup_name=startup_name,
            overall_score=5.0,
            investment_recommendation="Hold - Analysis Needed",
            
            problem_clarity=DimensionScore(
                name="Problem Clarity",
                score=5.0,
                reasoning="Evaluation processing error - manual review needed",
                suggestions=["Re-evaluate pitch manually"]
            ),
            
            market_opportunity=DimensionScore(
                name="Market Opportunity",
                score=5.0,
                reasoning="Evaluation processing error - manual review needed",
                suggestions=["Research market conditions"]
            ),
            
            business_model=DimensionScore(
                name="Business Model",
                score=5.0,
                reasoning="Evaluation processing error - manual review needed",
                suggestions=["Validate business model"]
            ),
            
            competitive_advantage=DimensionScore(
                name="Competitive Advantage",
                score=5.0,
                reasoning="Evaluation processing error - manual review needed",
                suggestions=["Identify competitive advantages"]
            ),
            
            team_strength=DimensionScore(
                name="Team Strength",
                score=5.0,
                reasoning="Evaluation processing error - manual review needed",
                suggestions=["Assess team capabilities"]
            ),
            
            key_strengths=["Manual evaluation needed"],
            main_concerns=["Processing error occurred"],
            next_steps=["Re-run evaluation", "Manual review", "Check system logs"]
        )


# Simple Chain-based Evaluator (alternative approach)
class SimpleChainEvaluator:
    """Simpler chain-based evaluator for comparison"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0.3)
        self._setup_chains()
    
    def _setup_chains(self):
        """Setup LangChain sequential chains"""
        
        # Step 1: Extract key information
        extraction_prompt = PromptTemplate(
            input_variables=["pitch"],
            template="""
            Extract key information from this startup pitch:
            {pitch}
            
            Extract and format as JSON:
            - Industry
            - Business model
            - Target market
            - Problem being solved
            - Solution proposed
            - Revenue model
            - Team size (if mentioned)
            """
        )
        
        self.extraction_chain = LLMChain(
            llm=self.llm,
            prompt=extraction_prompt,
            output_key="extracted_info"
        )
        
        # Step 2: Evaluate dimensions
        evaluation_prompt = PromptTemplate(
            input_variables=["pitch", "extracted_info"],
            template="""
            Based on this startup pitch and extracted information, provide evaluation scores:
            
            Pitch: {pitch}
            Key Info: {extracted_info}
            
            Score each dimension from 0-10 with reasoning:
            
            1. PROBLEM CLARITY: How well-defined is the problem?
            2. MARKET OPPORTUNITY: How large and accessible is the market?
            3. BUSINESS MODEL: How viable is the revenue model?
            4. COMPETITIVE ADVANTAGE: How defensible is the position?
            5. TEAM STRENGTH: How capable does the team appear?
            
            Provide overall investment recommendation and key insights.
            Format as structured text with clear sections.
            """
        )
        
        self.evaluation_chain = LLMChain(
            llm=self.llm,
            prompt=evaluation_prompt,
            output_key="evaluation"
        )
        
        # Combine into sequential chain
        self.full_chain = SequentialChain(
            chains=[self.extraction_chain, self.evaluation_chain],
            input_variables=["pitch"],
            output_variables=["extracted_info", "evaluation"],
            verbose=True
        )
    
    async def evaluate_pitch(self, pitch_text: str, startup_name: str) -> Dict[str, Any]:
        """Evaluate pitch using sequential chains"""
        result = await self.full_chain.ainvoke({"pitch": pitch_text})
        
        return {
            "startup_name": startup_name,
            "extracted_info": result["extracted_info"],
            "evaluation": result["evaluation"],
            "method": "chain_based"
        }


# Sample pitch data for testing
SAMPLE_PITCHES = [
    {
        "name": "HealthAI",
        "pitch": "HealthAI uses machine learning to analyze medical images and detect early signs of diseases. Our AI has achieved 94% accuracy in detecting pneumonia from chest X-rays. We're targeting rural hospitals that lack radiology specialists. Our SaaS model charges $0.50 per scan with pilot programs in 5 hospitals. The medical imaging market is $4.2B and growing at 15% annually. We're seeking $2M to scale our technology and expand to 50 hospitals."
    },
    {
        "name": "EduVerse",
        "pitch": "EduVerse creates immersive VR learning experiences for K-12 students. Our platform includes 100+ VR lessons across science, history, and math. We've improved test scores by 25% in pilot schools. The VR education market is projected to reach $13B by 2025. We charge schools $30/student/year and have partnerships with 15 schools. Seeking $1M to develop more content and expand our sales team."
    },
    {
        "name": "GreenCoin",
        "pitch": "GreenCoin is a blockchain platform that rewards users for sustainable behaviors like recycling and using public transport. Users earn tokens that can be redeemed for discounts at partner businesses. We have 10,000 active users and partnerships with 50 local businesses. The carbon credit market is worth $1B annually. Our revenue model includes transaction fees and corporate partnerships. Raising $500K to expand to new cities."
    }
]


# Testing and demonstration
async def test_evaluators():
    """Test both evaluation approaches"""
    
    print("🚀 Testing PitchPal AI Evaluators")
    print("=" * 50)
    
    # Initialize evaluators
    agent_evaluator = EvaluationAgent()
    chain_evaluator = SimpleChainEvaluator()
    
    # Test with sample pitch
    sample_pitch = SAMPLE_PITCHES[0]
    print(f"\n📊 Evaluating: {sample_pitch['name']}")
    print("-" * 30)
    
    try:
        # Test agent-based evaluation
        print("\n🤖 Agent-based Evaluation:")
        agent_result = await agent_evaluator.evaluate_pitch(
            sample_pitch['pitch'], 
            sample_pitch['name']
        )
        
        print(f"Overall Score: {agent_result.overall_score}/10")
        print(f"Recommendation: {agent_result.investment_recommendation}")
        print(f"Key Strengths: {', '.join(agent_result.key_strengths[:2])}")
        
        # Test chain-based evaluation
        print("\n⛓️ Chain-based Evaluation:")
        chain_result = await chain_evaluator.evaluate_pitch(
            sample_pitch['pitch'], 
            sample_pitch['name']
        )
        
        print("Extraction successful:", "✅" if chain_result['extracted_info'] else "❌")
        print("Evaluation completed:", "✅" if chain_result['evaluation'] else "❌")
        
        return agent_result, chain_result
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        return None, None


if __name__ == "__main__":
    # Run the test
    asyncio.run(test_evaluators())
