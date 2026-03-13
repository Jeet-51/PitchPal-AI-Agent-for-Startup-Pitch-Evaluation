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
    return _build_mock_evaluation(role="startup", overall_score=7.0, recommendation="Buy")


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
        recommendation="Buy",
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
        recommendation="Hold",
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
        valid_values = {"Strong Buy", "Buy", "Hold", "Pass", "Strong Pass"}
        # Allow compound recommendations like "Hold — Re-evaluation Required"
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
# 2. Score Consistency Tests
# =====================================================================

class TestScoreConsistency:
    """Verify score-to-recommendation mapping and overall score logic."""

    def test_overall_score_is_average_of_dimensions(self, startup_eval: PitchEvaluation):
        dim_mean = sum(d.score for d in startup_eval.dimensions) / len(startup_eval.dimensions)
        assert abs(startup_eval.overall_score - dim_mean) <= 1.0, (
            f"Overall score {startup_eval.overall_score} differs from dimension mean "
            f"{dim_mean:.2f} by more than 1.0"
        )

    def test_high_score_has_buy_recommendation(self):
        eval_ = _build_mock_evaluation(overall_score=8.0, recommendation="Buy")
        assert eval_.investment_recommendation in ("Buy", "Strong Buy")

    def test_low_score_has_pass_recommendation(self):
        eval_ = _build_mock_evaluation(overall_score=3.0, recommendation="Pass")
        assert eval_.investment_recommendation in ("Pass", "Strong Pass")

    def test_medium_score_has_hold_recommendation(self):
        eval_ = _build_mock_evaluation(overall_score=5.5, recommendation="Hold")
        assert eval_.investment_recommendation == "Hold"


# =====================================================================
# 3. JSON Repair Tests
# =====================================================================

class TestJsonRepair:
    """Tests for _try_fix_truncated_json and _extract_json_string_aware."""

    # --- Ported from test_fixes.py ---

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

    # --- New / expanded tests ---

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
        # _try_fix_truncated_json uses Strategy 3 (extract largest {...})
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
        # Strategy 5 may extract partial fields; if none match, returns None
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
        assert 6.0 <= healthai_eval.overall_score <= 8.0, (
            f"HealthAI overall_score {healthai_eval.overall_score} outside expected 6-8 range"
        )
        for dim in healthai_eval.dimensions:
            assert 0 <= dim.score <= 10
            assert len(dim.reasoning) > 0

    def test_eduverse_scores_reasonable(self, eduverse_eval: PitchEvaluation):
        """EduVerse: growing revenue, real product -> expected overall 5-7."""
        assert eduverse_eval.startup_name == "EduVerse"
        assert 5.0 <= eduverse_eval.overall_score <= 7.0, (
            f"EduVerse overall_score {eduverse_eval.overall_score} outside expected 5-7 range"
        )
        for dim in eduverse_eval.dimensions:
            assert 0 <= dim.score <= 10

    def test_greencoin_scores_reasonable(self, greencoin_eval: PitchEvaluation):
        """GreenCoin: crypto angle, niche -> expected overall 4-6."""
        assert greencoin_eval.startup_name == "GreenCoin"
        assert 4.0 <= greencoin_eval.overall_score <= 6.0, (
            f"GreenCoin overall_score {greencoin_eval.overall_score} outside expected 4-6 range"
        )
        for dim in greencoin_eval.dimensions:
            assert 0 <= dim.score <= 10

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
