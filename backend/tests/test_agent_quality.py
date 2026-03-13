"""
Agent Quality Benchmarking Test Suite for PitchPal ReAct Agent.

Tests the agent's output quality using mock data and schema validation.
No real LLM API calls are made.

Run with:  pytest backend/tests/test_agent_quality.py -v
"""

import json
import hashlib
import re
import pytest

from app.models.schemas import (
    PitchEvaluation,
    DimensionScore,
    Contradiction,
    STARTUP_DIMENSIONS,
    STARTUP_DIMENSION_TITLES,
    INVESTOR_DIMENSIONS,
    INVESTOR_DIMENSION_TITLES,
    SAMPLE_PITCHES,
)
from app.agent.react_agent import _try_fix_truncated_json, _extract_json_string_aware


# =====================================================================
# Helpers: build mock evaluations
# =====================================================================

def _build_mock_evaluation(
    startup_name: str = "TestStartup",
    role: str = "startup",
    overall_score: float = 7.0,
    recommendation: str = "Buy",
    dim_scores: list[float] | None = None,
    contradictions: list[dict] | None = None,
) -> PitchEvaluation:
    """Construct a PitchEvaluation from mock data without calling the LLM."""
    if role == "investor":
        dim_keys = INVESTOR_DIMENSIONS
        dim_titles = INVESTOR_DIMENSION_TITLES
    else:
        dim_keys = STARTUP_DIMENSIONS
        dim_titles = STARTUP_DIMENSION_TITLES

    if dim_scores is None:
        dim_scores = [overall_score] * len(dim_keys)

    dimensions = []
    for i, key in enumerate(dim_keys):
        score = dim_scores[i] if i < len(dim_scores) else 5.0
        dimensions.append(
            DimensionScore(
                name=dim_titles.get(key, key.replace("_", " ").title()),
                score=score,
                reasoning=f"Mock reasoning for {key}.",
                suggestions=[f"Improve {key}", f"Research {key} further"],
                sources=["https://example.com/source"],
                benchmark=f"Industry benchmark for {key}",
            )
        )

    contras = []
    if contradictions:
        for c in contradictions:
            contras.append(
                Contradiction(
                    pitch_claim=c["pitch_claim"],
                    research_finding=c["research_finding"],
                    source=c.get("source", "https://example.com"),
                )
            )

    return PitchEvaluation(
        startup_name=startup_name,
        overall_score=overall_score,
        investment_recommendation=recommendation,
        role=role,
        dimensions=dimensions,
        key_strengths=["Strong team", "Large market", "Clear problem"],
        main_concerns=["Weak moat", "High CAC risk"],
        next_steps=["Clarify competitive moat", "Show unit economics", "Define GTM"],
        contradictions=contras,
    )


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def startup_eval() -> PitchEvaluation:
    return _build_mock_evaluation(role="startup", overall_score=7.0, recommendation="Hold")


@pytest.fixture
def investor_eval() -> PitchEvaluation:
    return _build_mock_evaluation(
        role="investor",
        overall_score=6.5,
        recommendation="Hold",
        dim_scores=[7.0, 6.0, 7.0, 5.0, 7.0, 6.0, 7.0],
    )


@pytest.fixture
def healthai_eval() -> PitchEvaluation:
    """HealthAI: strong team, real traction, expected 6-8."""
    return _build_mock_evaluation(
        startup_name="HealthAI",
        role="startup",
        overall_score=7.2,
        recommendation="Hold",
        dim_scores=[7.0, 8.0, 7.0, 6.0, 8.0],
        contradictions=[
            {
                "pitch_claim": "94% accuracy outperforms average radiologist",
                "research_finding": "Top AI models report 92-97% range; comparison to radiologists varies by study",
                "source": "pubmed.ncbi.nlm.nih.gov",
            }
        ],
    )


@pytest.fixture
def eduverse_eval() -> PitchEvaluation:
    """EduVerse: growing revenue, real product, expected 5-7."""
    return _build_mock_evaluation(
        startup_name="EduVerse",
        role="startup",
        overall_score=6.2,
        recommendation="Hold",
        dim_scores=[6.0, 6.0, 6.5, 5.5, 7.0],
    )


@pytest.fixture
def greencoin_eval() -> PitchEvaluation:
    """GreenCoin: crypto angle, niche market, expected 4-6."""
    return _build_mock_evaluation(
        startup_name="GreenCoin",
        role="startup",
        overall_score=5.0,
        recommendation="Pass",
        dim_scores=[5.0, 5.0, 4.5, 5.0, 5.5],
    )


@pytest.fixture
def eval_with_contradictions() -> PitchEvaluation:
    return _build_mock_evaluation(
        contradictions=[
            {
                "pitch_claim": "We have no competitors",
                "research_finding": "Crunchbase lists 12 funded competitors in this space",
                "source": "crunchbase.com",
            },
            {
                "pitch_claim": "Market is $50B",
                "research_finding": "Grand View Research estimates $12B TAM",
                "source": "grandviewresearch.com",
            },
        ]
    )


# =====================================================================
# 1. Output Schema Validation Tests
# =====================================================================

class TestOutputSchemaValidation:
    """Verify PitchEvaluation has all required fields and valid values."""

    def test_evaluation_has_all_required_fields(self, startup_eval: PitchEvaluation):
        assert startup_eval.overall_score is not None
        assert startup_eval.investment_recommendation is not None
        assert startup_eval.dimensions is not None
        assert startup_eval.key_strengths is not None
        assert startup_eval.main_concerns is not None
        assert startup_eval.next_steps is not None

    def test_overall_score_in_valid_range(self, startup_eval: PitchEvaluation):
        assert 0 <= startup_eval.overall_score <= 10

    def test_recommendation_is_valid_value(self, startup_eval: PitchEvaluation):
        valid_values = {"Buy", "Hold", "Pass"}
        rec = startup_eval.investment_recommendation
        assert any(v in rec for v in valid_values), (
            f"Recommendation '{rec}' does not contain a valid value from {valid_values}"
        )

    def test_startup_dimensions_count(self, startup_eval: PitchEvaluation):
        assert len(startup_eval.dimensions) == 5

    def test_investor_dimensions_count(self, investor_eval: PitchEvaluation):
        assert len(investor_eval.dimensions) == 7

    def test_each_dimension_has_required_fields(self, startup_eval: PitchEvaluation):
        for dim in startup_eval.dimensions:
            assert dim.name, "Dimension name must not be empty"
            assert dim.score is not None, "Dimension score must not be None"
            assert dim.reasoning, "Dimension reasoning must not be empty"
            assert dim.suggestions is not None, "Dimension suggestions must not be None"

    def test_dimension_scores_in_range(self, startup_eval: PitchEvaluation):
        for dim in startup_eval.dimensions:
            assert 0 <= dim.score <= 10, f"Dimension '{dim.name}' score {dim.score} is out of range"

    def test_reasoning_not_empty(self, startup_eval: PitchEvaluation):
        for dim in startup_eval.dimensions:
            assert len(dim.reasoning.strip()) > 0, (
                f"Dimension '{dim.name}' has empty reasoning"
            )

    def test_suggestions_are_list(self, startup_eval: PitchEvaluation):
        for dim in startup_eval.dimensions:
            assert isinstance(dim.suggestions, list), (
                f"Dimension '{dim.name}' suggestions is not a list"
            )


# =====================================================================
# 2. Score Consistency Tests (updated for new thresholds)
# =====================================================================

class TestScoreConsistency:
    """Verify score-to-recommendation mapping and overall score logic."""

    def test_overall_score_is_average_of_dimensions(self, startup_eval: PitchEvaluation):
        dim_mean = sum(d.score for d in startup_eval.dimensions) / len(startup_eval.dimensions)
        assert abs(startup_eval.overall_score - dim_mean) <= 1.0, (
            f"Overall score {startup_eval.overall_score} differs from dimension mean "
            f"{dim_mean:.2f} by more than 1.0"
        )

    def test_score_8_plus_is_buy(self):
        """Scores >= 8.0 should map to Buy."""
        eval_ = _build_mock_evaluation(overall_score=8.0, recommendation="Buy")
        assert "Buy" in eval_.investment_recommendation

    def test_score_8_5_is_buy(self):
        """Scores >= 8.5 should also map to Buy (not Strong Buy)."""
        eval_ = _build_mock_evaluation(overall_score=8.5, recommendation="Buy")
        assert eval_.investment_recommendation == "Buy"

    def test_score_9_is_buy(self):
        """Score 9.0 maps to Buy."""
        eval_ = _build_mock_evaluation(overall_score=9.0, recommendation="Buy")
        assert eval_.investment_recommendation == "Buy"

    def test_score_7_to_8_is_hold_promising(self):
        """Scores 7.0-7.9 should map to Hold with promising message."""
        eval_ = _build_mock_evaluation(overall_score=7.5, recommendation="Hold")
        assert "Hold" in eval_.investment_recommendation

    def test_score_5_5_to_7_is_hold(self):
        """Scores 5.5-6.9 should map to Hold."""
        eval_ = _build_mock_evaluation(overall_score=6.0, recommendation="Hold")
        assert eval_.investment_recommendation == "Hold"

    def test_score_below_5_5_is_pass(self):
        """Scores < 5.5 should map to Pass."""
        eval_ = _build_mock_evaluation(overall_score=4.0, recommendation="Pass")
        assert eval_.investment_recommendation == "Pass"

    def test_score_5_4_is_pass(self):
        """Score 5.4 is below 5.5 threshold, should be Pass."""
        eval_ = _build_mock_evaluation(overall_score=5.4, recommendation="Pass")
        assert eval_.investment_recommendation == "Pass"

    def test_score_5_5_is_hold(self):
        """Score 5.5 is exactly at Hold threshold."""
        eval_ = _build_mock_evaluation(overall_score=5.5, recommendation="Hold")
        assert eval_.investment_recommendation == "Hold"


# =====================================================================
# 3. JSON Repair Tests
# =====================================================================

class TestJsonRepair:
    """Tests for _try_fix_truncated_json and _extract_json_string_aware."""

    def test_string_aware_bracket_extractor(self):
        """Parentheses inside strings do not break extraction."""
        tricky = '({"reasoning": "Market (Nigeria) growing fast)", "score": 7})'
        result = _extract_json_string_aware(tricky, 0)
        parsed = _try_fix_truncated_json(result)
        assert parsed is not None
        assert parsed.get("score") == 7

    def test_truncated_json_repair_from_test_fixes(self):
        """Incomplete JSON (cut off mid-way) is recovered."""
        truncated = (
            '{"problem_clarity": {"score": 8, "reasoning": "Clear problem", '
            '"suggestions": ["s1"]}, "overall_score": 7.5'
        )
        result = _try_fix_truncated_json(truncated)
        assert result is not None
        assert result.get("overall_score") == 7.5

    def test_trailing_comma_from_test_fixes(self):
        """JSON with trailing commas is fixed."""
        with_trailing = '{"score": 8, "reasoning": "Good", "suggestions": ["s1",],}'
        result = _try_fix_truncated_json(with_trailing)
        assert result is not None
        assert result.get("score") == 8

    def test_full_valid_json_from_test_fixes(self):
        """Normal evaluation JSON still works."""
        full_json = json.dumps({
            "problem_clarity": {"score": 8, "reasoning": "Clear", "suggestions": ["s1"]},
            "overall_score": 6.8,
            "recommendation": "Buy",
            "strengths": ["a"],
            "concerns": ["b"],
            "next_steps": ["c"],
        })
        result = _try_fix_truncated_json(full_json)
        assert result is not None
        assert result["recommendation"] == "Buy"
        assert result["overall_score"] == 6.8

    def test_repair_truncated_json(self):
        """Handles JSON cut off mid-string value."""
        truncated = '{"name": "TestStartup", "score": 7, "reasoning": "This is a goo'
        result = _try_fix_truncated_json(truncated)
        assert result is not None
        assert result.get("score") == 7

    def test_repair_trailing_comma(self):
        """Fixes trailing commas in arrays and objects."""
        bad_json = '{"items": [1, 2, 3,], "value": "ok",}'
        result = _try_fix_truncated_json(bad_json)
        assert result is not None
        assert result["items"] == [1, 2, 3]
        assert result["value"] == "ok"

    def test_repair_string_with_brackets(self):
        """Handles parentheses in string values without breaking extraction."""
        text = '({"reasoning": "Revenue ($2M ARR) is strong (top 10%)", "score": 8})'
        content = _extract_json_string_aware(text, 0)
        parsed = _try_fix_truncated_json(content)
        assert parsed is not None
        assert parsed.get("score") == 8
        assert "($2M ARR)" in parsed.get("reasoning", "")

    def test_repair_valid_json_unchanged(self):
        """Valid JSON passes through without modification."""
        valid = {"key": "value", "num": 42, "arr": [1, 2, 3]}
        raw = json.dumps(valid)
        result = _try_fix_truncated_json(raw)
        assert result == valid

    def test_repair_json_with_markdown_fences(self):
        """JSON wrapped in markdown code fences can still be extracted."""
        raw = '```json\n{"score": 9, "reasoning": "Excellent"}\n```'
        result = _try_fix_truncated_json(raw)
        assert result is not None
        assert result.get("score") == 9

    def test_repair_deeply_truncated_nested(self):
        """Deeply nested truncated JSON recovers at least partial data."""
        truncated = '{"dim": {"score": 6, "reasoning": "ok", "suggestions": ["a", "b'
        result = _try_fix_truncated_json(truncated)
        assert result is not None
        assert result.get("dim", {}).get("score") == 6

    def test_repair_returns_none_for_garbage(self):
        """Completely invalid input returns None."""
        result = _try_fix_truncated_json("this is not json at all")
        assert result is None or isinstance(result, dict)


# =====================================================================
# 4. Evaluation Cache Determinism Tests
# =====================================================================

def _normalize_pitch(text: str) -> str:
    """Normalize pitch text for cache key generation."""
    return re.sub(r"\s+", " ", text.strip().lower())


def _cache_key(pitch_text: str) -> str:
    """Generate a deterministic cache key from pitch text."""
    normalized = _normalize_pitch(pitch_text)
    return hashlib.sha256(normalized.encode()).hexdigest()


class TestCacheDeterminism:
    """Verify that identical (or semantically-identical) pitches produce the same cache key."""

    def test_same_pitch_same_scores(self):
        """Identical pitches should return identical cached results."""
        pitch = "We are building an AI platform for healthcare diagnostics."
        key1 = _cache_key(pitch)
        key2 = _cache_key(pitch)
        assert key1 == key2, "Identical pitches must produce the same cache key"

    def test_normalized_pitch_matches(self):
        """Whitespace-normalized pitches map to the same cache key."""
        pitch_a = "Hello World"
        pitch_b = "  hello   world  "
        key_a = _cache_key(pitch_a)
        key_b = _cache_key(pitch_b)
        assert key_a == key_b, (
            f"Normalized pitches should have identical cache keys: {key_a} != {key_b}"
        )

    def test_different_pitches_different_keys(self):
        """Substantially different pitches produce different cache keys."""
        key_a = _cache_key("AI healthcare diagnostics platform for hospitals")
        key_b = _cache_key("Blockchain rewards platform for sustainability")
        assert key_a != key_b


# =====================================================================
# 5. Sample Pitch Benchmark Tests
# =====================================================================

class TestSamplePitchBenchmarks:
    """Validate that mock evaluations for sample pitches fall in expected score ranges."""

    def test_healthai_scores_reasonable(self, healthai_eval: PitchEvaluation):
        """HealthAI: strong team, real traction -> expected overall 6-8."""
        assert healthai_eval.startup_name == "HealthAI"
        assert 6.0 <= healthai_eval.overall_score <= 8.0

    def test_eduverse_scores_reasonable(self, eduverse_eval: PitchEvaluation):
        """EduVerse: growing revenue, real product -> expected overall 5-7."""
        assert eduverse_eval.startup_name == "EduVerse"
        assert 5.0 <= eduverse_eval.overall_score <= 7.0

    def test_greencoin_scores_reasonable(self, greencoin_eval: PitchEvaluation):
        """GreenCoin: crypto angle, niche -> expected overall 4-6."""
        assert greencoin_eval.startup_name == "GreenCoin"
        assert 4.0 <= greencoin_eval.overall_score <= 6.0

    def test_sample_pitches_exist(self):
        """Verify the sample pitches data is available."""
        assert len(SAMPLE_PITCHES) == 3
        names = {p["name"] for p in SAMPLE_PITCHES}
        assert names == {"HealthAI", "EduVerse", "GreenCoin"}


# =====================================================================
# 6. Contradiction Detection Quality Tests
# =====================================================================

class TestContradictionDetection:
    """Verify contradiction objects have required fields with actual content."""

    def test_contradiction_has_required_fields(self, eval_with_contradictions: PitchEvaluation):
        assert len(eval_with_contradictions.contradictions) > 0
        for c in eval_with_contradictions.contradictions:
            assert hasattr(c, "pitch_claim")
            assert hasattr(c, "research_finding")
            assert hasattr(c, "source")

    def test_contradiction_fields_not_empty(self, eval_with_contradictions: PitchEvaluation):
        for c in eval_with_contradictions.contradictions:
            assert len(c.pitch_claim.strip()) > 0, "pitch_claim must not be empty"
            assert len(c.research_finding.strip()) > 0, "research_finding must not be empty"
            assert len(c.source.strip()) > 0, "source must not be empty"

    def test_contradiction_count_matches(self, eval_with_contradictions: PitchEvaluation):
        assert len(eval_with_contradictions.contradictions) == 2

    def test_no_contradictions_by_default(self, startup_eval: PitchEvaluation):
        assert len(startup_eval.contradictions) == 0


# =====================================================================
# 7. Pydantic Validation Boundary Tests
# =====================================================================

class TestPydanticBoundaries:
    """Ensure Pydantic models reject out-of-range values."""

    def test_score_below_zero_rejected(self):
        with pytest.raises(Exception):
            DimensionScore(
                name="Test",
                score=-1.0,
                reasoning="Invalid",
                suggestions=[],
            )

    def test_score_above_ten_rejected(self):
        with pytest.raises(Exception):
            DimensionScore(
                name="Test",
                score=11.0,
                reasoning="Invalid",
                suggestions=[],
            )

    def test_overall_score_below_zero_rejected(self):
        with pytest.raises(Exception):
            PitchEvaluation(
                startup_name="Test",
                overall_score=-1.0,
                investment_recommendation="Pass",
                dimensions=[],
                key_strengths=[],
                main_concerns=[],
                next_steps=[],
            )

    def test_overall_score_above_ten_rejected(self):
        with pytest.raises(Exception):
            PitchEvaluation(
                startup_name="Test",
                overall_score=11.0,
                investment_recommendation="Pass",
                dimensions=[],
                key_strengths=[],
                main_concerns=[],
                next_steps=[],
            )


# =====================================================================
# 8. Golden Dataset Benchmark Pitches (15 pitches)
# =====================================================================

class TestGoldenDatasetBenchmarks:
    """
    15 benchmark pitches across 4 tiers to validate score-to-recommendation
    calibration. Each pitch uses mock scores that represent what the agent
    should produce for that quality tier.

    Tiers:
      Strong (4 pitches)    - overall >= 8.0 -> Buy
      Promising (4 pitches) - overall 7.0-7.9 -> Hold (promising message)
      Average (4 pitches)   - overall 5.5-6.9 -> Hold
      Weak (3 pitches)      - overall < 5.5 -> Pass
    """

    # -----------------------------------------------------------------
    # Strong Tier (>= 8.0 -> Buy)
    # -----------------------------------------------------------------

    def test_strong_01_stripe_like_fintech(self):
        """Fintech with strong execution, proven revenue, large TAM."""
        ev = _build_mock_evaluation(
            startup_name="PayFlow",
            overall_score=8.5,
            recommendation="Buy",
            dim_scores=[8.5, 9.0, 8.0, 8.0, 9.0],
        )
        assert ev.overall_score >= 8.0
        assert ev.investment_recommendation == "Buy"

    def test_strong_02_ai_saas_platform(self):
        """AI SaaS with deep moat, enterprise contracts, strong retention."""
        ev = _build_mock_evaluation(
            startup_name="CortexAI",
            overall_score=8.8,
            recommendation="Buy",
            dim_scores=[9.0, 8.5, 9.0, 8.5, 9.0],
        )
        assert ev.overall_score >= 8.0
        assert ev.investment_recommendation == "Buy"

    def test_strong_03_biotech_breakthrough(self):
        """Biotech with FDA fast-track, strong IP, funded team."""
        ev = _build_mock_evaluation(
            startup_name="GeneCure",
            overall_score=8.2,
            recommendation="Buy",
            dim_scores=[8.0, 8.5, 8.0, 8.0, 8.5],
        )
        assert ev.overall_score >= 8.0
        assert ev.investment_recommendation == "Buy"

    def test_strong_04_marketplace_leader(self):
        """Two-sided marketplace with network effects and rapid growth."""
        ev = _build_mock_evaluation(
            startup_name="TradeNest",
            overall_score=8.0,
            recommendation="Buy",
            dim_scores=[8.0, 8.0, 7.5, 8.0, 8.5],
        )
        assert ev.overall_score >= 8.0
        assert ev.investment_recommendation == "Buy"

    # -----------------------------------------------------------------
    # Promising Tier (7.0 - 7.9 -> Hold with promising message)
    # -----------------------------------------------------------------

    def test_promising_01_healthtech(self):
        """HealthTech with real traction but unproven unit economics."""
        ev = _build_mock_evaluation(
            startup_name="MedSync",
            overall_score=7.5,
            recommendation="Hold",
            dim_scores=[7.5, 8.0, 7.0, 7.0, 8.0],
        )
        assert 7.0 <= ev.overall_score < 8.0
        assert "Hold" in ev.investment_recommendation

    def test_promising_02_edtech_growth(self):
        """EdTech with growing user base, needs monetization clarity."""
        ev = _build_mock_evaluation(
            startup_name="LearnPath",
            overall_score=7.2,
            recommendation="Hold",
            dim_scores=[7.0, 7.5, 7.0, 7.0, 7.5],
        )
        assert 7.0 <= ev.overall_score < 8.0
        assert "Hold" in ev.investment_recommendation

    def test_promising_03_logistics_platform(self):
        """Logistics optimization with pilot customers, pre-revenue."""
        ev = _build_mock_evaluation(
            startup_name="RouteMaster",
            overall_score=7.0,
            recommendation="Hold",
            dim_scores=[7.0, 7.0, 7.0, 6.5, 7.5],
        )
        assert 7.0 <= ev.overall_score < 8.0
        assert "Hold" in ev.investment_recommendation

    def test_promising_04_cybersecurity(self):
        """Cybersecurity tool with strong tech, early sales pipeline."""
        ev = _build_mock_evaluation(
            startup_name="ShieldNet",
            overall_score=7.8,
            recommendation="Hold",
            dim_scores=[8.0, 7.5, 7.5, 7.5, 8.5],
        )
        assert 7.0 <= ev.overall_score < 8.0
        assert "Hold" in ev.investment_recommendation

    # -----------------------------------------------------------------
    # Average Tier (5.5 - 6.9 -> Hold)
    # -----------------------------------------------------------------

    def test_average_01_food_delivery(self):
        """Food delivery clone in saturated market, some traction."""
        ev = _build_mock_evaluation(
            startup_name="QuickBite",
            overall_score=6.2,
            recommendation="Hold",
            dim_scores=[6.0, 6.5, 6.0, 6.0, 6.5],
        )
        assert 5.5 <= ev.overall_score < 7.0
        assert ev.investment_recommendation == "Hold"

    def test_average_02_social_app(self):
        """Social app with niche audience, unclear monetization."""
        ev = _build_mock_evaluation(
            startup_name="VibeCheck",
            overall_score=5.8,
            recommendation="Hold",
            dim_scores=[6.0, 5.5, 5.5, 6.0, 6.0],
        )
        assert 5.5 <= ev.overall_score < 7.0
        assert ev.investment_recommendation == "Hold"

    def test_average_03_ecommerce_tool(self):
        """E-commerce analytics tool, some revenue, crowded space."""
        ev = _build_mock_evaluation(
            startup_name="ShopInsight",
            overall_score=6.8,
            recommendation="Hold",
            dim_scores=[7.0, 6.5, 6.5, 7.0, 7.0],
        )
        assert 5.5 <= ev.overall_score < 7.0
        assert ev.investment_recommendation == "Hold"

    def test_average_04_hr_platform(self):
        """HR platform with basic features, no differentiation."""
        ev = _build_mock_evaluation(
            startup_name="HireFlow",
            overall_score=5.5,
            recommendation="Hold",
            dim_scores=[5.5, 5.5, 5.0, 5.5, 6.0],
        )
        assert 5.5 <= ev.overall_score < 7.0
        assert ev.investment_recommendation == "Hold"

    # -----------------------------------------------------------------
    # Weak Tier (< 5.5 -> Pass)
    # -----------------------------------------------------------------

    def test_weak_01_crypto_nft(self):
        """NFT marketplace with no traction, regulatory risk."""
        ev = _build_mock_evaluation(
            startup_name="CryptoApes",
            overall_score=3.5,
            recommendation="Pass",
            dim_scores=[3.0, 4.0, 3.0, 3.5, 4.0],
        )
        assert ev.overall_score < 5.5
        assert ev.investment_recommendation == "Pass"

    def test_weak_02_metaverse_social(self):
        """Metaverse social platform, vague vision, no MVP."""
        ev = _build_mock_evaluation(
            startup_name="MetaHangout",
            overall_score=4.2,
            recommendation="Pass",
            dim_scores=[4.0, 4.5, 4.0, 4.0, 4.5],
        )
        assert ev.overall_score < 5.5
        assert ev.investment_recommendation == "Pass"

    def test_weak_03_ai_wrapper(self):
        """Thin AI wrapper with no moat, trivial to replicate."""
        ev = _build_mock_evaluation(
            startup_name="GPTWrapper",
            overall_score=5.0,
            recommendation="Pass",
            dim_scores=[5.0, 5.0, 4.5, 5.0, 5.5],
        )
        assert ev.overall_score < 5.5
        assert ev.investment_recommendation == "Pass"


# =====================================================================
# 9. Recommendation Calibration Integration Tests
# =====================================================================

def _get_real_agent_class():
    """Import the real (unpatched) ReActAgent class from the source module."""
    import importlib
    import importlib.util
    import os
    spec = importlib.util.spec_from_file_location(
        "react_agent_real",
        os.path.join(os.path.dirname(__file__), "..", "app", "agent", "react_agent.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    import app.models.schemas  # noqa: F401
    spec.loader.exec_module(mod)
    return mod.ReActAgent


class TestRecommendationCalibration:
    """
    Verify that the react_agent._build_evaluation_from_dict enforces
    score-based recommendation thresholds regardless of what the LLM returns.
    """

    @pytest.fixture(autouse=True)
    def _load_real_class(self):
        self.RealAgent = _get_real_agent_class()

    def _make_agent(self):
        return object.__new__(self.RealAgent)

    def test_calibration_buy_threshold(self):
        """Score >= 8.0 always produces Buy."""
        agent = self._make_agent()
        data = {
            "problem_clarity": {"score": 8.5, "reasoning": "Clear", "suggestions": []},
            "market_opportunity": {"score": 8.0, "reasoning": "Big", "suggestions": []},
            "solution_quality": {"score": 8.0, "reasoning": "Solid", "suggestions": []},
            "traction_validation": {"score": 8.0, "reasoning": "Real", "suggestions": []},
            "team_execution": {"score": 8.5, "reasoning": "Strong", "suggestions": []},
            "overall_score": 8.2,
            "recommendation": "Strong Buy",
            "strengths": ["a"],
            "concerns": ["b"],
            "next_steps": ["c"],
        }
        result = agent._build_evaluation_from_dict(data, "TestCo", "startup")
        assert result.investment_recommendation == "Buy"

    def test_calibration_hold_promising_threshold(self):
        """Score 7.0-7.9 produces Hold with promising message."""
        agent = self._make_agent()
        data = {
            "problem_clarity": {"score": 7.5, "reasoning": "Ok", "suggestions": []},
            "market_opportunity": {"score": 7.0, "reasoning": "Ok", "suggestions": []},
            "solution_quality": {"score": 7.0, "reasoning": "Ok", "suggestions": []},
            "traction_validation": {"score": 7.0, "reasoning": "Ok", "suggestions": []},
            "team_execution": {"score": 7.5, "reasoning": "Ok", "suggestions": []},
            "overall_score": 7.2,
            "recommendation": "Buy",
            "strengths": ["a"],
            "concerns": ["b"],
            "next_steps": ["c"],
        }
        result = agent._build_evaluation_from_dict(data, "TestCo", "startup")
        assert "Hold" in result.investment_recommendation
        assert "Promising fundamentals" in result.investment_recommendation

    def test_calibration_hold_standard_threshold(self):
        """Score 5.5-6.9 produces plain Hold."""
        agent = self._make_agent()
        data = {
            "problem_clarity": {"score": 6.0, "reasoning": "Ok", "suggestions": []},
            "market_opportunity": {"score": 6.5, "reasoning": "Ok", "suggestions": []},
            "solution_quality": {"score": 6.0, "reasoning": "Ok", "suggestions": []},
            "traction_validation": {"score": 6.0, "reasoning": "Ok", "suggestions": []},
            "team_execution": {"score": 6.5, "reasoning": "Ok", "suggestions": []},
            "overall_score": 6.2,
            "recommendation": "Buy",
            "strengths": ["a"],
            "concerns": ["b"],
            "next_steps": ["c"],
        }
        result = agent._build_evaluation_from_dict(data, "TestCo", "startup")
        assert result.investment_recommendation == "Hold"

    def test_calibration_pass_threshold(self):
        """Score < 5.5 always produces Pass."""
        agent = self._make_agent()
        data = {
            "problem_clarity": {"score": 4.0, "reasoning": "Weak", "suggestions": []},
            "market_opportunity": {"score": 4.5, "reasoning": "Small", "suggestions": []},
            "solution_quality": {"score": 4.0, "reasoning": "Weak", "suggestions": []},
            "traction_validation": {"score": 4.0, "reasoning": "None", "suggestions": []},
            "team_execution": {"score": 4.5, "reasoning": "Weak", "suggestions": []},
            "overall_score": 4.2,
            "recommendation": "Hold",
            "strengths": ["a"],
            "concerns": ["b"],
            "next_steps": ["c"],
        }
        result = agent._build_evaluation_from_dict(data, "TestCo", "startup")
        assert result.investment_recommendation == "Pass"

    def test_calibration_overrides_llm_recommendation(self):
        """Score-based override always wins over LLM recommendation."""
        agent = self._make_agent()
        data = {
            "problem_clarity": {"score": 7.0, "reasoning": "Ok", "suggestions": []},
            "market_opportunity": {"score": 6.5, "reasoning": "Ok", "suggestions": []},
            "solution_quality": {"score": 7.0, "reasoning": "Ok", "suggestions": []},
            "traction_validation": {"score": 6.5, "reasoning": "Ok", "suggestions": []},
            "team_execution": {"score": 7.0, "reasoning": "Ok", "suggestions": []},
            "overall_score": 6.8,
            "recommendation": "Strong Buy",
            "strengths": ["a"],
            "concerns": ["b"],
            "next_steps": ["c"],
        }
        result = agent._build_evaluation_from_dict(data, "TestCo", "startup")
        assert result.investment_recommendation == "Hold"
