"""
Quick test to verify the 3 bug fixes in react_agent.py
Run: python test_fixes.py
"""
import sys
sys.path.insert(0, ".")

from app.agent.react_agent import _extract_json_string_aware, _try_fix_truncated_json

print("=" * 55)
print("PitchPal Bug Fix Verification")
print("=" * 55)

# ── Test 1: String-aware bracket extractor ─────────────────
print("\n[1] String-aware bracket extractor")
print("    Checks: parentheses inside strings don't break extraction")

tricky = '({"reasoning": "Market (Nigeria) growing fast)", "score": 7})'
result = _extract_json_string_aware(tricky, 0)
parsed = _try_fix_truncated_json(result)

if parsed and parsed.get("score") == 7:
    print("    ✅ PASS — extracted correctly despite () in string")
else:
    print(f"    ❌ FAIL — result: {result[:80]}")

# ── Test 2: Truncated JSON repair ──────────────────────────
print("\n[2] Truncated JSON repair")
print("    Checks: incomplete JSON (cut off mid-way) is recovered")

truncated = '{"problem_clarity": {"score": 8, "reasoning": "Clear problem", "suggestions": ["s1"]}, "overall_score": 7.5'
# Missing closing }
result = _try_fix_truncated_json(truncated)

if result and result.get("overall_score") == 7.5:
    print("    ✅ PASS — truncated JSON repaired successfully")
else:
    print(f"    ❌ FAIL — result: {result}")

# ── Test 3: Trailing comma repair ─────────────────────────
print("\n[3] Trailing comma repair")
print("    Checks: JSON with trailing commas is fixed")

with_trailing = '{"score": 8, "reasoning": "Good", "suggestions": ["s1",],}'
result = _try_fix_truncated_json(with_trailing)

if result and result.get("score") == 8:
    print("    ✅ PASS — trailing commas cleaned")
else:
    print(f"    ❌ FAIL — result: {result}")

# ── Test 4: Full valid JSON parses correctly ───────────────
print("\n[4] Full valid JSON parsing")
print("    Checks: normal evaluation JSON still works")

full_json = '''{
  "problem_clarity": {"score": 8, "reasoning": "Clear problem statement", "suggestions": ["Quantify the pain point"]},
  "market_opportunity": {"score": 7, "reasoning": "Large TAM", "suggestions": ["Add SAM/SOM breakdown"]},
  "business_model": {"score": 6, "reasoning": "Unit economics need clarity", "suggestions": ["Show CAC/LTV"]},
  "competitive_advantage": {"score": 5, "reasoning": "Moat unclear", "suggestions": ["Define IP strategy"]},
  "team_strength": {"score": 8, "reasoning": "Strong founding team", "suggestions": ["Add advisor board"]},
  "overall_score": 6.8,
  "recommendation": "Buy",
  "strengths": ["Strong team", "Large market", "Clear problem"],
  "concerns": ["Weak moat", "High CAC risk"],
  "next_steps": ["Clarify competitive moat", "Show unit economics", "Define go-to-market"]
}'''

result = _try_fix_truncated_json(full_json)
if result and result.get("recommendation") == "Buy" and result.get("overall_score") == 6.8:
    print("    ✅ PASS — valid JSON parsed correctly")
else:
    print(f"    ❌ FAIL — result: {result}")

# ── Summary ───────────────────────────────────────────────
print("\n" + "=" * 55)
print("All tests complete. ✅ = working, ❌ = needs attention")
print("=" * 55)
