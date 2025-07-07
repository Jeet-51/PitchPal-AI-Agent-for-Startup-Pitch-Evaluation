"""
FastAPI server with LangServe integration for PitchPal
Demonstrates production API deployment of LangChain chains
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import asyncio
import uvicorn
import os
from datetime import datetime

from langserve import add_routes
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from pitch_evaluator import EvaluationAgent, SimpleChainEvaluator


# Pydantic models for API
class PitchInput(BaseModel):
    startup_name: str
    pitch_text: str
    evaluator_type: str = "agent"  # "agent" or "chain"


class EvaluationResponse(BaseModel):
    startup_name: str
    overall_score: Optional[float] = None
    investment_recommendation: Optional[str] = None
    method: str
    evaluation_data: Dict[str, Any]
    timestamp: datetime
    processing_time: float


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str
    uptime: str


# Initialize FastAPI app
app = FastAPI(
    title="PitchPal API",
    description="AI-powered startup pitch evaluation API using LangChain and OpenAI",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
start_time = datetime.now()
evaluation_count = 0


# Simple LangChain chain for LangServe demo
def create_simple_evaluation_chain():
    """Create a simple evaluation chain for LangServe"""
    prompt = ChatPromptTemplate.from_template("""
    You are an expert startup evaluator. Analyze this pitch and provide a brief evaluation:
    
    Startup: {startup_name}
    Pitch: {pitch_text}
    
    Provide:
    1. Overall score (0-10)
    2. Investment recommendation (Buy/Hold/Pass)
    3. Key strengths (2-3 points)
    4. Main concerns (2-3 points)
    
    Keep response concise and structured.
    """)
    
    llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain


# Add LangServe routes
try:
    evaluation_chain = create_simple_evaluation_chain()
    add_routes(
        app,
        evaluation_chain,
        path="/langserve/evaluate",
        enable_feedback_endpoint=True,
        enable_public_trace_link_endpoint=True,
        playground_type="default"
    )
except Exception as e:
    print(f"Warning: LangServe setup failed: {e}")


# API Routes
@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with API health info"""
    uptime = datetime.now() - start_time
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0",
        uptime=str(uptime).split('.')[0]  # Remove microseconds
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    uptime = datetime.now() - start_time
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0",
        uptime=str(uptime).split('.')[0]
    )


@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_pitch(pitch_input: PitchInput):
    """Evaluate a startup pitch using AI agents or chains"""
    global evaluation_count
    
    start_eval_time = datetime.now()
    
    try:
        if pitch_input.evaluator_type == "agent":
            # Use AI Agent evaluator
            evaluator = EvaluationAgent()
            result = await evaluator.evaluate_pitch(
                pitch_input.pitch_text, 
                pitch_input.startup_name
            )
            
            evaluation_data = {
                "overall_score": result.overall_score,
                "investment_recommendation": result.investment_recommendation,
                "dimensions": {
                    "problem_clarity": {
                        "score": result.problem_clarity.score,
                        "reasoning": result.problem_clarity.reasoning
                    },
                    "market_opportunity": {
                        "score": result.market_opportunity.score,
                        "reasoning": result.market_opportunity.reasoning
                    },
                    "business_model": {
                        "score": result.business_model.score,
                        "reasoning": result.business_model.reasoning
                    },
                    "competitive_advantage": {
                        "score": result.competitive_advantage.score,
                        "reasoning": result.competitive_advantage.reasoning
                    },
                    "team_strength": {
                        "score": result.team_strength.score,
                        "reasoning": result.team_strength.reasoning
                    }
                },
                "insights": {
                    "key_strengths": result.key_strengths,
                    "main_concerns": result.main_concerns,
                    "next_steps": result.next_steps
                }
            }
            
            response = EvaluationResponse(
                startup_name=result.startup_name,
                overall_score=result.overall_score,
                investment_recommendation=result.investment_recommendation,
                method="AI Agent (ReAct)",
                evaluation_data=evaluation_data,
                timestamp=datetime.now(),
                processing_time=(datetime.now() - start_eval_time).total_seconds()
            )
            
        else:
            # Use Sequential Chain evaluator
            evaluator = SimpleChainEvaluator()
            result = await evaluator.evaluate_pitch(
                pitch_input.pitch_text, 
                pitch_input.startup_name
            )
            
            evaluation_data = {
                "extracted_info": result.get("extracted_info", ""),
                "evaluation": result.get("evaluation", ""),
                "method": result.get("method", "chain_based")
            }
            
            response = EvaluationResponse(
                startup_name=result["startup_name"],
                overall_score=None,  # Chain method doesn't provide structured scores
                investment_recommendation=None,
                method="Sequential Chains",
                evaluation_data=evaluation_data,
                timestamp=datetime.now(),
                processing_time=(datetime.now() - start_eval_time).total_seconds()
            )
        
        evaluation_count += 1
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@app.get("/stats")
async def get_stats():
    """Get API usage statistics"""
    uptime = datetime.now() - start_time
    
    return {
        "total_evaluations": evaluation_count,
        "uptime_seconds": uptime.total_seconds(),
        "uptime_formatted": str(uptime).split('.')[0],
        "start_time": start_time.isoformat(),
        "current_time": datetime.now().isoformat(),
        "status": "operational"
    }


@app.get("/sample-pitches")
async def get_sample_pitches():
    """Get sample startup pitches for testing"""
    from pitch_evaluator import SAMPLE_PITCHES
    return {"sample_pitches": SAMPLE_PITCHES}


# Batch evaluation endpoint
@app.post("/batch-evaluate")
async def batch_evaluate(pitches: List[PitchInput]):
    """Evaluate multiple pitches in parallel"""
    if len(pitches) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 pitches per batch")
    
    start_time = datetime.now()
    
    try:
        # Create evaluation tasks
        tasks = []
        for pitch in pitches:
            if pitch.evaluator_type == "agent":
                evaluator = EvaluationAgent()
                task = evaluator.evaluate_pitch(pitch.pitch_text, pitch.startup_name)
            else:
                evaluator = SimpleChainEvaluator()
                task = evaluator.evaluate_pitch(pitch.pitch_text, pitch.startup_name)
            tasks.append(task)
        
        # Run evaluations in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        evaluations = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                evaluations.append({
                    "startup_name": pitches[i].startup_name,
                    "error": str(result),
                    "success": False
                })
            else:
                if hasattr(result, 'overall_score'):
                    # Agent result
                    evaluations.append({
                        "startup_name": result.startup_name,
                        "overall_score": result.overall_score,
                        "investment_recommendation": result.investment_recommendation,
                        "method": "AI Agent",
                        "success": True
                    })
                else:
                    # Chain result
                    evaluations.append({
                        "startup_name": result["startup_name"],
                        "method": "Sequential Chains",
                        "evaluation_summary": result["evaluation"][:200] + "...",
                        "success": True
                    })
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "batch_size": len(pitches),
            "successful_evaluations": sum(1 for e in evaluations if e.get("success", False)),
            "failed_evaluations": sum(1 for e in evaluations if not e.get("success", False)),
            "processing_time": processing_time,
            "results": evaluations
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch evaluation failed: {str(e)}")


# WebSocket endpoint for real-time evaluation
@app.websocket("/ws/evaluate")
async def websocket_evaluate(websocket):
    """WebSocket endpoint for real-time pitch evaluation"""
    await websocket.accept()
    
    try:
        while True:
            # Receive pitch data
            data = await websocket.receive_json()
            startup_name = data.get("startup_name")
            pitch_text = data.get("pitch_text")
            evaluator_type = data.get("evaluator_type", "agent")
            
            if not startup_name or not pitch_text:
                await websocket.send_json({
                    "error": "Missing startup_name or pitch_text"
                })
                continue
            
            # Send processing status
            await websocket.send_json({
                "status": "processing",
                "message": f"Evaluating {startup_name}..."
            })
            
            # Run evaluation
            try:
                if evaluator_type == "agent":
                    evaluator = EvaluationAgent()
                    result = await evaluator.evaluate_pitch(pitch_text, startup_name)
                    
                    await websocket.send_json({
                        "status": "completed",
                        "startup_name": result.startup_name,
                        "overall_score": result.overall_score,
                        "investment_recommendation": result.investment_recommendation,
                        "key_strengths": result.key_strengths[:3],
                        "main_concerns": result.main_concerns[:3]
                    })
                else:
                    evaluator = SimpleChainEvaluator()
                    result = await evaluator.evaluate_pitch(pitch_text, startup_name)
                    
                    await websocket.send_json({
                        "status": "completed",
                        "startup_name": result["startup_name"],
                        "method": "Sequential Chains",
                        "evaluation_preview": result["evaluation"][:300] + "..."
                    })
                    
            except Exception as e:
                await websocket.send_json({
                    "status": "error",
                    "error": str(e)
                })
                
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()


# Custom exception handler
@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {"error": "Internal server error", "detail": str(exc)}


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize app on startup"""
    print("🚀 PitchPal API Server Starting...")
    print(f"📊 Docs available at: http://localhost:8000/docs")
    print(f"🔗 LangServe playground at: http://localhost:8000/langserve/evaluate/playground")


if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )