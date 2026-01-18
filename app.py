"""
PitchPal Streamlit App - Portfolio Demo
Showcases LangChain, tools, async workflows, and OpenAI integration
"""

import streamlit as st
import asyncio
import json
import os
for k in ["OPENAI_PROXY", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]:
    os.environ.pop(k, None)
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

    class SimplePandas:
        @staticmethod
        def DataFrame(data):
            return data

    pd = SimplePandas()

from pitch_evaluator import EvaluationAgent, SimpleChainEvaluator, SAMPLE_PITCHES


# -----------------------------
# Helpers
# -----------------------------
def run_async(coro):
    """
    Run async code safely in Streamlit.
    Avoids common event loop issues across environments.
    """
    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            # Streamlit sometimes already has a loop, so create a new loop for this call
            new_loop = asyncio.new_event_loop()
            try:
                return new_loop.run_until_complete(coro)
            finally:
                new_loop.close()
    except RuntimeError:
        pass

    return asyncio.run(coro)


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="PitchPal - AI Startup Evaluator",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown(
    """
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
""",
    unsafe_allow_html=True
)

# -----------------------------
# Session state
# -----------------------------
if "evaluation_history" not in st.session_state:
    st.session_state.evaluation_history = []
if "current_evaluation" not in st.session_state:
    st.session_state.current_evaluation = None


def get_score_color_class(score):
    if score >= 7:
        return "score-high"
    if score >= 5:
        return "score-medium"
    return "score-low"


def get_recommendation_class(recommendation):
    if "buy" in recommendation.lower():
        return "recommendation-buy"
    if "hold" in recommendation.lower():
        return "recommendation-hold"
    return "recommendation-pass"


def create_radar_chart(evaluation):
    dimensions = [
        evaluation.problem_clarity.name,
        evaluation.market_opportunity.name,
        evaluation.business_model.name,
        evaluation.competitive_advantage.name,
        evaluation.team_strength.name,
    ]

    scores = [
        evaluation.problem_clarity.score,
        evaluation.market_opportunity.score,
        evaluation.business_model.score,
        evaluation.competitive_advantage.score,
        evaluation.team_strength.score,
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=scores,
            theta=dimensions,
            fill="toself",
            name=evaluation.startup_name,
        )
    )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        showlegend=False,
        title=f"{evaluation.startup_name} - Evaluation Radar",
        font=dict(size=14),
    )
    return fig


def create_score_breakdown_chart(evaluation):
    dimensions = [
        "Problem Clarity",
        "Market Opportunity",
        "Business Model",
        "Competitive Advantage",
        "Team Strength",
    ]

    scores = [
        evaluation.problem_clarity.score,
        evaluation.market_opportunity.score,
        evaluation.business_model.score,
        evaluation.competitive_advantage.score,
        evaluation.team_strength.score,
    ]

    fig = go.Figure(
        go.Bar(
            x=scores,
            y=dimensions,
            orientation="h",
            text=[f"{s:.1f}" for s in scores],
            textposition="inside",
        )
    )

    fig.update_layout(
        title="Score Breakdown by Dimension",
        xaxis_title="Score (0-10)",
        xaxis=dict(range=[0, 10]),
        height=400,
        font=dict(size=12),
    )
    return fig


async def run_evaluation(pitch_text, startup_name, evaluator_type):
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


def display_agent_results(evaluation):
    col1, col2 = st.columns(2)

    with col1:
        score_class = get_score_color_class(evaluation.overall_score)
        st.markdown(
            f"""
        <div class="dimension-card">
            <h3 style="color: #333; margin-bottom: 10px;">Overall Score</h3>
            <div class="{score_class}" style="font-size: 2rem;">
                {evaluation.overall_score:.1f}/10
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        rec_class = get_recommendation_class(evaluation.investment_recommendation)
        st.markdown(
            f"""
        <div class="dimension-card">
            <h3 style="color: #333; margin-bottom: 10px;">Investment Recommendation</h3>
            <div class="{rec_class}">
                {evaluation.investment_recommendation}
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.plotly_chart(create_radar_chart(evaluation), use_container_width=True)
    st.plotly_chart(create_score_breakdown_chart(evaluation), use_container_width=True)

    st.subheader("🔍 Detailed Analysis")
    dimensions = [
        evaluation.problem_clarity,
        evaluation.market_opportunity,
        evaluation.business_model,
        evaluation.competitive_advantage,
        evaluation.team_strength,
    ]

    for dim in dimensions:
        with st.expander(f"{dim.name} - {dim.score:.1f}/10"):
            st.write(f"**Reasoning:** {dim.reasoning}")
            if dim.suggestions:
                st.write("**Suggestions:**")
                for suggestion in dim.suggestions:
                    st.write(f"• {suggestion}")

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
    st.subheader("📋 Extracted Information")
    st.code(result["extracted_info"], language="json")

    st.subheader("📊 Evaluation Analysis")
    st.write(result["evaluation"])

    st.info(
        "This is a text-based evaluation from the Sequential Chains (LCEL) approach. "
        "For structured scoring and charts, use the AI Agent (ReAct) method."
    )


def main():
    st.markdown(
        '<h1 class="main-header">🚀 PitchPal - AI Startup Evaluator</h1>',
        unsafe_allow_html=True,
    )
    st.markdown("### Powered by LangChain, Tools & OpenAI")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        # API key: Streamlit Cloud secrets OR manual input
        api_key_configured = False
        try:
            if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
                st.success("✅ OpenAI API Key Configured")
                os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
                api_key_configured = True
            else:
                raise KeyError("No secrets configured")
        except Exception:
            api_key = st.text_input(
                "OpenAI API Key",
                type="password",
                help="Enter your OpenAI API key to enable evaluations",
            )
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
                api_key_configured = True

        st.divider()

        evaluator_type = st.selectbox(
            "🤖 Choose Evaluator",
            ["AI Agent (ReAct)", "Sequential Chains (LCEL)"],
            help="Select the AI approach for evaluation",
        )

        if evaluator_type == "AI Agent (ReAct)":
            st.info(
                """
🤖 **AI Agent (ReAct Method)**
- Dynamic reasoning with tool-assisted analysis
- Uses custom tools: market research, competitor analysis, financial modeling
- Outputs structured scores (0-10) with insights and visualizations
"""
            )
        else:
            st.info(
                """
⛓️ **Sequential Chains (LCEL)**
- Linear processing: Extract -> Evaluate
- No legacy LLMChain/SequentialChain dependencies
- Outputs text-based analysis for comparison
"""
            )

        st.divider()

        st.header("📝 Sample Pitches")
        for i, pitch in enumerate(SAMPLE_PITCHES):
            if st.button(f"📊 {pitch['name']}", key=f"sample_{i}"):
                st.session_state.selected_pitch = pitch

        st.divider()

        st.header("📈 History")
        if st.session_state.evaluation_history:
            for i, eval_data in enumerate(st.session_state.evaluation_history[-3:]):
                score = eval_data.get("score")
                score_text = f"{score:.1f}/10" if isinstance(score, (int, float)) else "N/A"

                with st.expander(f"{eval_data['name']} - {score_text}"):
                    st.write(f"**Method:** {eval_data['method']}")
                    st.write(f"**Date:** {eval_data['timestamp']}")
        else:
            st.write("No evaluations yet")

    # Main content
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📝 Pitch Input")

        startup_name = st.text_input("🏢 Startup Name", placeholder="Enter startup name")

        if "selected_pitch" in st.session_state:
            pitch_text = st.text_area(
                "📋 Pitch Description",
                value=st.session_state.selected_pitch["pitch"],
                height=300,
                help="Enter the startup pitch description",
            )
            if not startup_name:
                startup_name = st.session_state.selected_pitch["name"]
                st.rerun()
        else:
            pitch_text = st.text_area(
                "📋 Pitch Description",
                height=300,
                placeholder="Enter a detailed startup pitch...",
                help="Enter the startup pitch description",
            )

        if st.button(
            "🚀 Evaluate Pitch",
            type="primary",
            disabled=not (startup_name and pitch_text and api_key_configured),
        ):
            if not api_key_configured:
                st.error("Please enter your OpenAI API key in the sidebar")
            elif not startup_name or not pitch_text:
                st.error("Please provide both startup name and pitch description")
            else:
                with st.spinner(f"🤖 Analyzing with {evaluator_type}..."):
                    result, method = run_async(run_evaluation(pitch_text, startup_name, evaluator_type))

                    if result:
                        st.session_state.current_evaluation = result

                        if method == "agent":
                            st.session_state.evaluation_history.append(
                                {
                                    "name": result.startup_name,
                                    "score": result.overall_score,
                                    "method": evaluator_type,
                                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                                    "evaluation": result,
                                }
                            )
                        else:
                            st.session_state.evaluation_history.append(
                                {
                                    "name": result["startup_name"],
                                    "score": None,
                                    "method": evaluator_type,
                                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                                    "evaluation": result,
                                }
                            )

                        st.success("✅ Evaluation completed!")
                        st.rerun()

    with col2:
        st.header("📊 Evaluation Results")

        if st.session_state.current_evaluation:
            result = st.session_state.current_evaluation
            if hasattr(result, "overall_score"):
                display_agent_results(result)
            else:
                display_chain_results(result)
        else:
            st.info("👆 Enter a pitch and click 'Evaluate Pitch' to see results")

            st.subheader("📈 Sample Evaluation")
            demo_data = {
                "Dimension": [
                    "Problem Clarity",
                    "Market Opportunity",
                    "Business Model",
                    "Competitive Advantage",
                    "Team Strength",
                ],
                "Score": [8.2, 7.5, 6.8, 7.1, 8.0],
            }

            if PANDAS_AVAILABLE:
                df = pd.DataFrame(demo_data)
                fig = px.bar(
                    df,
                    x="Score",
                    y="Dimension",
                    orientation="h",
                    title="Demo Evaluation Scores",
                )
            else:
                fig = go.Figure(
                    go.Bar(
                        x=demo_data["Score"],
                        y=demo_data["Dimension"],
                        orientation="h",
                        text=[f"{s:.1f}" for s in demo_data["Score"]],
                        textposition="inside",
                    )
                )
                fig.update_layout(title="Demo Evaluation Scores")

            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)


def show_analytics():
    st.header("📈 Analytics Dashboard")

    if not st.session_state.evaluation_history:
        st.info("No evaluation data available. Complete some evaluations first!")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Evaluations", len(st.session_state.evaluation_history))

    with col2:
        agent_evals = sum(1 for e in st.session_state.evaluation_history if "Agent" in e["method"])
        st.metric("Agent Evaluations", agent_evals)

    with col3:
        chain_evals = sum(1 for e in st.session_state.evaluation_history if "Sequential" in e["method"])
        st.metric("Chain Evaluations", chain_evals)

    agent_evaluations = [e for e in st.session_state.evaluation_history if isinstance(e.get("score"), (int, float))]

    if agent_evaluations:
        scores = [e["score"] for e in agent_evaluations]
        names = [e["name"] for e in agent_evaluations]

        fig = px.bar(x=names, y=scores, title="Evaluation Scores by Startup")
        fig.update_layout(xaxis_title="Startup", yaxis_title="Overall Score")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📊 Score Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Average Score", f"{sum(scores) / len(scores):.1f}")
        with col2:
            st.metric("Highest Score", f"{max(scores):.1f}")
        with col3:
            st.metric("Lowest Score", f"{min(scores):.1f}")
        with col4:
            strong_pitches = sum(1 for s in scores if s >= 7.5)
            st.metric("Strong Pitches (≥7.5)", strong_pitches)


def show_about():
    st.header("ℹ️ About PitchPal")

    st.markdown(
        """
## 🚀 What is PitchPal?

PitchPal is an AI-powered startup pitch evaluation platform built to demonstrate modern LangChain patterns, tool integration, and OpenAI model usage.

## 🛠️ Technologies Used

- **LangChain**: Prompts, tools, and LCEL chains
- **OpenAI**: Chat models for analysis
- **Streamlit**: Interactive UI
- **Pydantic**: Structured output and validation
- **Plotly**: Visualizations

## 🤖 Evaluation Methods

### 1) AI Evaluator (Tool-based)
- Extracts key details
- Uses tools for market/competitor/financial analysis
- Produces structured scoring (0-10) with recommendations

### 2) Sequential Chains (LCEL)
- Extract -> Evaluate in a fixed pipeline
- Returns a text-based analysis for comparison
"""
    )


def app_navigation():
    st.sidebar.header("🧭 Navigation")
    pages = {
        "🏠 Home": "main",
        "📈 Analytics": "analytics",
        "ℹ️ About": "about",
    }
    selected_page = st.sidebar.selectbox("Choose Page", list(pages.keys()))
    return pages[selected_page]


if __name__ == "__main__":
    current_page = app_navigation()

    if current_page == "main":
        main()
    elif current_page == "analytics":
        show_analytics()
    elif current_page == "about":
        show_about()

    st.sidebar.markdown("---")
    st.sidebar.markdown("**PitchPal v1.0**")
    st.sidebar.markdown("Portfolio Project")
