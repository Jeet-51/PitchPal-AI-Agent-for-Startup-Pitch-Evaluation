"""
PitchPal Streamlit App - Portfolio Demo
Showcases LangChain, AI Agents, and OpenAI integration
"""

import streamlit as st
import asyncio
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import streamlit as st
import os

# Configure API key from Streamlit secrets or user input
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    default_key = "✅ API Key Configured"
    disabled = True
else:
    default_key = ""
    disabled = False

# API Key input in sidebar
api_key = st.text_input(
    "OpenAI API Key", 
    value=default_key,
    type="password" if default_key == "" else "default",
    disabled=disabled,
    help="Enter your OpenAI API key or use the pre-configured one"
)

if api_key and api_key != "✅ API Key Configured":
    os.environ["OPENAI_API_KEY"] = api_key
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    # Create a simple substitute for basic DataFrame functionality
    class SimplePandas:
        @staticmethod
        def DataFrame(data):
            return data
    pd = SimplePandas()

from pitch_evaluator import EvaluationAgent, SimpleChainEvaluator, SAMPLE_PITCHES


# Page configuration
st.set_page_config(
    page_title="PitchPal - AI Startup Evaluator",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}

.dimension-card {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 10px;
    border-left: 4px solid #1f77b4;
    margin: 0.5rem 0;
    color: #333;
}

.dimension-card h3 {
    color: #333 !important;
    margin-bottom: 10px;
}

.score-high {
    color: #28a745 !important;
    font-weight: bold;
}

.score-medium {
    color: #ffc107 !important;
    font-weight: bold;
}

.score-low {
    color: #dc3545 !important;
    font-weight: bold;
}

.recommendation-buy {
    background-color: #d4edda;
    color: #155724 !important;
    padding: 0.5rem;
    border-radius: 5px;
    font-weight: bold;
}

.recommendation-hold {
    background-color: #fff3cd;
    color: #856404 !important;
    padding: 0.5rem;
    border-radius: 5px;
    font-weight: bold;
}

.recommendation-pass {
    background-color: #f8d7da;
    color: #721c24 !important;
    padding: 0.5rem;
    border-radius: 5px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'evaluation_history' not in st.session_state:
    st.session_state.evaluation_history = []
if 'current_evaluation' not in st.session_state:
    st.session_state.current_evaluation = None


def get_score_color_class(score):
    """Get CSS class based on score"""
    if score >= 7:
        return "score-high"
    elif score >= 5:
        return "score-medium"
    else:
        return "score-low"


def get_recommendation_class(recommendation):
    """Get CSS class based on recommendation"""
    if "buy" in recommendation.lower():
        return "recommendation-buy"
    elif "hold" in recommendation.lower():
        return "recommendation-hold"
    else:
        return "recommendation-pass"


def create_radar_chart(evaluation):
    """Create radar chart for evaluation dimensions"""
    dimensions = [
        evaluation.problem_clarity.name,
        evaluation.market_opportunity.name,
        evaluation.business_model.name,
        evaluation.competitive_advantage.name,
        evaluation.team_strength.name
    ]
    
    scores = [
        evaluation.problem_clarity.score,
        evaluation.market_opportunity.score,
        evaluation.business_model.score,
        evaluation.competitive_advantage.score,
        evaluation.team_strength.score
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=scores,
        theta=dimensions,
        fill='toself',
        name=evaluation.startup_name,
        line_color='rgba(31, 119, 180, 0.8)',
        fillcolor='rgba(31, 119, 180, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10]
            )),
        showlegend=False,
        title=f"{evaluation.startup_name} - Evaluation Radar",
        font=dict(size=14)
    )
    
    return fig


def create_score_breakdown_chart(evaluation):
    """Create horizontal bar chart for score breakdown"""
    dimensions = ["Problem Clarity", "Market Opportunity", "Business Model", "Competitive Advantage", "Team Strength"]
    scores = [
        evaluation.problem_clarity.score,
        evaluation.market_opportunity.score,
        evaluation.business_model.score,
        evaluation.competitive_advantage.score,
        evaluation.team_strength.score
    ]
    
    colors = ['#28a745' if s >= 7 else '#ffc107' if s >= 5 else '#dc3545' for s in scores]
    
    fig = go.Figure(go.Bar(
        x=scores,
        y=dimensions,
        orientation='h',
        marker_color=colors,
        text=[f"{s:.1f}" for s in scores],
        textposition='inside'
    ))
    
    fig.update_layout(
        title="Score Breakdown by Dimension",
        xaxis_title="Score (0-10)",
        xaxis=dict(range=[0, 10]),
        height=400,
        font=dict(size=12)
    )
    
    return fig


async def run_evaluation(pitch_text, startup_name, evaluator_type):
    """Run pitch evaluation"""
    try:
        if evaluator_type == "AI Agent (ReAct)":
            evaluator = EvaluationAgent()
            result = await evaluator.evaluate_pitch(pitch_text, startup_name)
            return result, "agent"
        else:
            evaluator = SimpleChainEvaluator()
            result = await evaluator.evaluate_pitch(pitch_text, startup_name)
            return result, "chain"
    except Exception as e:
        st.error(f"Evaluation failed: {str(e)}")
        return None, None


def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown('<h1 class="main-header">🚀 PitchPal - AI Startup Evaluator</h1>', unsafe_allow_html=True)
    st.markdown("### Powered by LangChain, AI Agents & OpenAI")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input("OpenAI API Key", type="password", 
                               help="Enter your OpenAI API key to enable evaluations")
        
        if api_key:
            import os
            os.environ["OPENAI_API_KEY"] = api_key
        
        st.divider()
        
        # Evaluator selection
        evaluator_type = st.selectbox(
            "🤖 Choose Evaluator",
            ["AI Agent (ReAct)", "Sequential Chains"],
            help="Select the AI approach for evaluation"
        )
        
        # Add explanations for each method
        if evaluator_type == "AI Agent (ReAct)":
            st.info("""
            🤖 **AI Agent (ReAct Method)**
            - **Dynamic decision-making**: Agent decides which tools to use
            - **Uses custom tools**: Market research, competitor analysis, financial modeling
            - **Reasoning + Acting**: Thinks through each step and takes actions
            - **Structured output**: Provides scores (0-10) for each dimension
            - **Advanced LangChain**: Demonstrates sophisticated agent capabilities
            """)
        else:
            st.info("""
            ⛓️ **Sequential Chains**
            - **Linear processing**: Fixed sequence of steps (Extract → Analyze → Evaluate)
            - **No tools**: Direct LLM processing without external functions
            - **Predictable workflow**: Same process for every pitch
            - **Text-based output**: Provides detailed written analysis
            - **Core LangChain**: Demonstrates fundamental chain concepts
            """)
        
        st.divider()
        
        # Sample pitches
        st.header("📝 Sample Pitches")
        for i, pitch in enumerate(SAMPLE_PITCHES):
            if st.button(f"📊 {pitch['name']}", key=f"sample_{i}"):
                st.session_state.selected_pitch = pitch
        
        st.divider()
        
        # Evaluation history
        st.header("📈 History")
        if st.session_state.evaluation_history:
            for i, eval_data in enumerate(st.session_state.evaluation_history[-3:]):
                with st.expander(f"{eval_data['name']} - {eval_data['score']:.1f}/10"):
                    st.write(f"**Method:** {eval_data['method']}")
                    st.write(f"**Date:** {eval_data['timestamp']}")
        else:
            st.write("No evaluations yet")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Pitch Input")
        
        # Startup name input
        startup_name = st.text_input("🏢 Startup Name", placeholder="Enter startup name")
        
        # Handle sample pitch selection
        if 'selected_pitch' in st.session_state:
            pitch_text = st.text_area(
                "📋 Pitch Description", 
                value=st.session_state.selected_pitch['pitch'],
                height=300,
                help="Enter the startup pitch description"
            )
            if not startup_name:
                startup_name = st.session_state.selected_pitch['name']
                st.rerun()
        else:
            pitch_text = st.text_area(
                "📋 Pitch Description", 
                height=300,
                placeholder="Enter a detailed startup pitch...",
                help="Enter the startup pitch description"
            )
        
        # Evaluate button
        if st.button("🚀 Evaluate Pitch", type="primary", disabled=not (startup_name and pitch_text and api_key)):
            if not api_key:
                st.error("Please enter your OpenAI API key in the sidebar")
            elif not startup_name or not pitch_text:
                st.error("Please provide both startup name and pitch description")
            else:
                with st.spinner(f"🤖 Analyzing with {evaluator_type}..."):
                    # Run evaluation
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    result, method = loop.run_until_complete(
                        run_evaluation(pitch_text, startup_name, evaluator_type)
                    )
                    
                    if result:
                        st.session_state.current_evaluation = result
                        
                        # Add to history
                        if method == "agent":
                            st.session_state.evaluation_history.append({
                                'name': result.startup_name,
                                'score': result.overall_score,
                                'method': evaluator_type,
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M"),
                                'evaluation': result
                            })
                        else:
                            st.session_state.evaluation_history.append({
                                'name': result['startup_name'],
                                'score': 7.5,  # Placeholder for chain method
                                'method': evaluator_type,
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M"),
                                'evaluation': result
                            })
                        
                        st.success("✅ Evaluation completed!")
                        st.rerun()
    
    with col2:
        st.header("📊 Evaluation Results")
        
        if st.session_state.current_evaluation:
            result = st.session_state.current_evaluation
            
            # Check if it's agent-based or chain-based result
            if hasattr(result, 'overall_score'):
                # Agent-based evaluation
                display_agent_results(result)
            else:
                # Chain-based evaluation
                display_chain_results(result)
        else:
            st.info("👆 Enter a pitch and click 'Evaluate Pitch' to see results")
            
            # Show demo visualization
            st.subheader("📈 Sample Evaluation")
            demo_data = {
                'Dimension': ['Problem Clarity', 'Market Opportunity', 'Business Model', 'Competitive Advantage', 'Team Strength'],
                'Score': [8.2, 7.5, 6.8, 7.1, 8.0]
            }
            
            # Create chart with or without pandas
            if PANDAS_AVAILABLE:
                df = pd.DataFrame(demo_data)
                fig = px.bar(df, x='Score', y='Dimension', orientation='h', 
                            color='Score', color_continuous_scale='RdYlGn',
                            title="Demo Evaluation Scores")
            else:
                # Use plotly directly without pandas
                fig = go.Figure(go.Bar(
                    x=demo_data['Score'],
                    y=demo_data['Dimension'],
                    orientation='h',
                    marker_color=demo_data['Score'],
                    text=[f"{s:.1f}" for s in demo_data['Score']],
                    textposition='inside'
                ))
                fig.update_layout(title="Demo Evaluation Scores")
                
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)


def display_agent_results(evaluation):
    """Display results from agent-based evaluation"""
    
    # Overall score and recommendation
    col1, col2 = st.columns(2)
    
    with col1:
        score_class = get_score_color_class(evaluation.overall_score)
        st.markdown(f"""
        <div class="dimension-card">
            <h3 style="color: #333; margin-bottom: 10px;">Overall Score</h3>
            <div class="{score_class}" style="font-size: 2rem;">
                {evaluation.overall_score:.1f}/10
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        rec_class = get_recommendation_class(evaluation.investment_recommendation)
        st.markdown(f"""
        <div class="dimension-card">
            <h3 style="color: #333; margin-bottom: 10px;">Investment Recommendation</h3>
            <div class="{rec_class}">
                {evaluation.investment_recommendation}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Radar chart
    st.plotly_chart(create_radar_chart(evaluation), use_container_width=True)
    
    # Score breakdown
    st.plotly_chart(create_score_breakdown_chart(evaluation), use_container_width=True)
    
    # Detailed dimension analysis
    st.subheader("🔍 Detailed Analysis")
    
    dimensions = [
        evaluation.problem_clarity,
        evaluation.market_opportunity,
        evaluation.business_model,
        evaluation.competitive_advantage,
        evaluation.team_strength
    ]
    
    for dim in dimensions:
        with st.expander(f"{dim.name} - {dim.score:.1f}/10"):
            st.write(f"**Reasoning:** {dim.reasoning}")
            if dim.suggestions:
                st.write("**Suggestions:**")
                for suggestion in dim.suggestions:
                    st.write(f"• {suggestion}")
    
    # Key insights
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("💪 Key Strengths")
        for strength in evaluation.key_strengths:
            st.write(f"✅ {strength}")
    
    with col2:
        st.subheader("⚠️ Main Concerns")
        for concern in evaluation.main_concerns:
            st.write(f"⚠️ {concern}")
    
    with col3:
        st.subheader("🎯 Next Steps")
        for step in evaluation.next_steps:
            st.write(f"🎯 {step}")


def display_chain_results(result):
    """Display results from chain-based evaluation"""
    
    st.subheader("📋 Extracted Information")
    st.code(result['extracted_info'], language='json')
    
    st.subheader("📊 Evaluation Analysis")
    st.write(result['evaluation'])
    
    st.info("💡 This is a text-based evaluation from the Sequential Chains approach. For structured scoring, use the AI Agent (ReAct) method.")


# Analytics page
def show_analytics():
    """Show evaluation analytics"""
    st.header("📈 Analytics Dashboard")
    
    if not st.session_state.evaluation_history:
        st.info("No evaluation data available. Complete some evaluations first!")
        return
    
    # Evaluation summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Evaluations", len(st.session_state.evaluation_history))
    
    with col2:
        agent_evals = sum(1 for e in st.session_state.evaluation_history if "Agent" in e['method'])
        st.metric("Agent Evaluations", agent_evals)
    
    with col3:
        chain_evals = sum(1 for e in st.session_state.evaluation_history if "Chain" in e['method'])
        st.metric("Chain Evaluations", chain_evals)
    
    # Score distribution
    agent_evaluations = [e for e in st.session_state.evaluation_history if "Agent" in e['method']]
    
    if agent_evaluations:
        scores = [e['score'] for e in agent_evaluations]
        names = [e['name'] for e in agent_evaluations]
        
        fig = px.bar(x=names, y=scores, title="Evaluation Scores by Startup",
                    color=scores, color_continuous_scale='RdYlGn')
        fig.update_layout(xaxis_title="Startup", yaxis_title="Overall Score")
        st.plotly_chart(fig, use_container_width=True)
        
        # Score statistics
        st.subheader("📊 Score Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Average Score", f"{sum(scores)/len(scores):.1f}")
        with col2:
            st.metric("Highest Score", f"{max(scores):.1f}")
        with col3:
            st.metric("Lowest Score", f"{min(scores):.1f}")
        with col4:
            strong_pitches = sum(1 for s in scores if s >= 7.5)
            st.metric("Strong Pitches (≥7.5)", strong_pitches)


# Navigation
def app_navigation():
    """App navigation"""
    st.sidebar.header("🧭 Navigation")
    
    pages = {
        "🏠 Home": "main",
        "📈 Analytics": "analytics",
        "ℹ️ About": "about"
    }
    
    selected_page = st.sidebar.selectbox("Choose Page", list(pages.keys()))
    return pages[selected_page]


def show_about():
    """Show about page"""
    st.header("ℹ️ About PitchPal")
    
    st.markdown("""
    ## 🚀 What is PitchPal?
    
    PitchPal is an AI-powered startup pitch evaluation platform built to demonstrate advanced LangChain capabilities, AI agents, and OpenAI integration.
    
    ## 🛠️ Technologies Used
    
    - **LangChain**: Framework for building LLM applications
    - **OpenAI GPT-4**: Primary language model for analysis
    - **AI Agents**: ReAct agents with custom tools
    - **Sequential Chains**: Multi-step evaluation pipeline
    - **Streamlit**: Interactive web interface
    - **Plotly**: Data visualizations
    
    ## 🤖 Evaluation Methods
    
    ### 1. AI Agent (ReAct) - **Recommended**
    - **What it is**: Advanced LangChain agents that reason and act dynamically
    - **How it works**: 
      - 🧠 **Thinks**: "I need market research for this pitch"
      - 🛠️ **Acts**: Calls MarketResearchTool to get industry data
      - 🧠 **Thinks**: "Now I need competitor analysis"
      - 🛠️ **Acts**: Calls CompetitorAnalysisTool
      - 🧠 **Thinks**: "Let me analyze the financials"
      - 🛠️ **Acts**: Calls FinancialModelingTool
      - 📊 **Decides**: Creates structured evaluation with 0-10 scores
    - **Output**: Structured scores, radar charts, detailed insights
    - **Best for**: Comprehensive analysis with visual results
    
    ### 2. Sequential Chains - **Educational**
    - **What it is**: Linear LangChain processing in fixed steps
    - **How it works**:
      - 📝 **Step 1**: Extract key information from pitch
      - 🔍 **Step 2**: Analyze the extracted data
      - 📋 **Step 3**: Generate written evaluation
    - **Output**: Text-based analysis and insights
    - **Best for**: Understanding core LangChain concepts
    
    ## 🆚 **Comparison Table**
    
    | Feature | AI Agent (ReAct) | Sequential Chains |
    |---------|------------------|-------------------|
    | **Complexity** | Advanced | Beginner-friendly |
    | **Decision Making** | Dynamic | Fixed sequence |
    | **Tool Usage** | ✅ Uses 3 custom tools | ❌ No tools |
    | **Output Format** | Structured scores (0-10) | Text analysis |
    | **Visualizations** | ✅ Radar charts, graphs | ❌ Text only |
    | **LangChain Concepts** | Agents, Tools, ReAct | Chains, Prompts |
    | **Processing Time** | Longer (more thorough) | Faster (simpler) |
    | **Portfolio Impact** | 🚀 Shows advanced skills | 📚 Shows fundamentals |
    
    ## 📊 Evaluation Dimensions
    
    1. **Problem Clarity** - How well-defined is the problem?
    2. **Market Opportunity** - Size and accessibility of the market
    3. **Business Model** - Viability of the revenue model
    4. **Competitive Advantage** - Strength of differentiation
    5. **Team Strength** - Capability of the founding team
    
    ## 🎯 Portfolio Purpose
    
    This project showcases:
    - LangChain framework mastery
    - AI agent implementation
    - Tool creation and integration
    - Chain orchestration
    - Production-ready UI development
    
    ---
    
    **Built with ❤️ for portfolio demonstration**
    """)


# Main app execution
if __name__ == "__main__":
    # Navigation
    current_page = app_navigation()
    
    if current_page == "main":
        main()
    elif current_page == "analytics":
        show_analytics()
    elif current_page == "about":
        show_about()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**PitchPal v1.0**")
    st.sidebar.markdown("Portfolio Project")