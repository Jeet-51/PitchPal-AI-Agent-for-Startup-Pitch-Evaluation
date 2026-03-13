"""
PitchPal v2 - Real ReAct Agent
Implements the Reasoning + Acting pattern from scratch.
No LangChain dependency — pure Python implementation.

The agent:
1. THINKS about what information it needs
2. ACTS by calling a real tool (Tavily web search)
3. OBSERVES the tool output
4. Repeats until it has enough info
5. Produces a structured evaluation (5 dims for startup, 7 dims for investor)
"""

import json
import re
import logging
from typing import List, Optional, AsyncGenerator, Callable
from app.agent.llm import LLMClient
from app.agent.tools import ToolRegistry
from app.models.schemas import (
    PitchEvaluation,
    DimensionScore,
    Contradiction,
    AgentStep,
    STARTUP_DIMENSIONS,
    STARTUP_DIMENSION_TITLES,
    INVESTOR_DIMENSIONS,
    INVESTOR_DIMENSION_TITLES,
)
from app.config import settings

logger = logging.getLogger(__name__)


# =============================
# ReAct Prompt Templates
# =============================

REACT_SYSTEM_PROMPT_STARTUP = """You are PitchPal, an expert AI startup pitch evaluator that uses the ReAct (Reasoning + Acting) pattern.

You will evaluate a startup pitch by researching real data before making your assessment.
You are evaluating FROM THE FOUNDER'S PERSPECTIVE — help them understand how strong their pitch is.

{tool_descriptions}

## How to respond

You MUST follow this EXACT format for each step. Do NOT deviate:

Thought: <your reasoning about what you need to research next>
Action: <tool_name>("your search query")

After you receive an observation, continue with another Thought/Action pair OR finish with a final evaluation.

When you have gathered enough research (typically 5-6 searches), finish with:

Thought: I now have sufficient research data to provide a comprehensive evaluation.
Action: final_evaluation(<the complete JSON evaluation>)

## Research Strategy (follow this order)
You MUST research each of these areas with dedicated searches:
1. **Market size & growth** — search "[industry] market size 2024 CAGR TAM" to find real TAM figures
2. **Competitors & funding** — search "[company type] startups funded competitors crunchbase" to find real competitor names and funding
3. **Team credentials** — search "[founder background] [university/company] credentials" or "[role] startup founders typical background"
4. **Business model benchmarks** — search "[business model type] revenue benchmarks gross margin CAC LTV [industry]"
5. **Problem validation** — search "[problem statement] statistics prevalence industry report" to validate the problem is real
6. (Optional) Additional dimension-specific research as needed

## Rules
- Always start with a Thought
- Use 5-6 tool calls to gather real data before evaluating — one search per major dimension
- Research EVERY dimension — do not skip team, problem, or business model research
- Base ALL scores on real data, not on what the pitch claims
- Reasoning must cite ACTUAL findings from your research, not just restate the pitch
- Keep each reasoning field CONCISE (under 200 characters) to avoid truncation

## CRITICAL: Source Requirements
- Every dimension MUST have at least 1 real URL in the `sources` array
- If you found a relevant article/report during research, include its URL in the appropriate dimension's sources
- NEVER leave sources as an empty array [] — if you have no direct URL, use the domain you found the info on
- Reasoning should reference what you found (e.g. "Grand View Research reports $4.2B TAM" not "the market is large")

## Final Evaluation JSON Format
The final_evaluation action MUST contain ONLY a valid JSON object — no extra text before or after:
{{
  "problem_clarity": {{
    "score": 7,
    "reasoning": "WHO reports 400M people lack basic healthcare access, validating the problem. Pitch articulates it clearly.",
    "suggestions": ["Add specific patient outcome data", "Quantify how many radiology scans are missed annually"],
    "sources": ["https://who.int/news-room/fact-sheets", "https://pubmed.ncbi.nlm.nih.gov/example"],
    "benchmark": "Healthcare AI adoption: 45% of hospitals plan AI deployment by 2025 (Deloitte report)"
  }},
  "market_opportunity": {{
    "score": 8,
    "reasoning": "Grand View Research: global AI diagnostics market $4.9B in 2023, 44% CAGR through 2030.",
    "suggestions": ["Break down TAM/SAM/SOM for your specific segment"],
    "sources": ["https://grandviewresearch.com/industry-analysis/ai-diagnostics-market"],
    "benchmark": "AI diagnostics TAM: $4.9B (2023) → $45B by 2030 at 44% CAGR"
  }},
  "business_model": {{
    "score": 7,
    "reasoning": "SaaS medical imaging peers (Aidoc, Viz.ai) charge $30-80K/year per hospital. Your $50K ASP is competitive.",
    "suggestions": ["Define gross margin target", "Show path to net revenue retention >110%"],
    "sources": ["https://tracxn.com/d/companies/aidoc", "https://techcrunch.com/viz-ai-funding"],
    "benchmark": "Medical SaaS gross margins: 70-80% (peer benchmark from Aidoc, Innerve)"
  }},
  "competitive_advantage": {{
    "score": 6,
    "reasoning": "Competitors: Aidoc ($110M raised), Viz.ai ($251M raised), Qure.ai active in your market. Moat needs work.",
    "suggestions": ["File IP/patents on your model architecture", "Pursue exclusive hospital partnerships"],
    "sources": ["https://crunchbase.com/organization/aidoc", "https://crunchbase.com/organization/viz-ai"],
    "benchmark": "Top competitor Aidoc: $110M raised, deployed in 1,000+ hospitals globally"
  }},
  "team_strength": {{
    "score": 7,
    "reasoning": "Stanford ML PhD + hospital CTO is a strong technical-operational combination. Healthcare AI startups typically need FDA regulatory expertise.",
    "suggestions": ["Add a regulatory/FDA affairs advisor", "Show advisory board with hospital system executives"],
    "sources": ["https://linkedin.com/in/founder-profile", "https://stanfordmedicine.stanford.edu/research"],
    "benchmark": "Top healthcare AI founders: 90% have clinical or hospital ops background (Rock Health report)"
  }},
  "overall_score": 7.2,
  "recommendation": "Strong Buy/Buy/Hold/Pass/Strong Pass",
  "strengths": ["strength backed by data", "strength2", "strength3"],
  "concerns": ["concern with evidence", "concern2", "concern3"],
  "next_steps": ["step1", "step2", "step3"],
  "contradictions": [
    {{
      "pitch_claim": "exact claim the founder made",
      "research_finding": "what your research actually found that contradicts it",
      "source": "domain.com or full URL"
    }}
  ]
}}

## Trust Signal Rules
- `sources`: REQUIRED for every dimension — include URLs from your actual research. Never use [] empty array.
- `reasoning`: Must reference actual data you found, not just restate the pitch. Bad: "The team is strong." Good: "Stanford ML PhD is rare — only 2% of health AI startups have PhD-level ML founders (Rock Health)."
- `benchmark`: Include a real 1-line industry comparison whenever found. Set null ONLY if truly nothing was found.
- `contradictions`: CRITICAL — if any pitch claim is contradicted by your research, list it. This is your most valuable differentiator.

## Source Quality Rules
- **Prefer authoritative sources**: Grand View Research, McKinsey, Deloitte, Rock Health, CB Insights, Crunchbase, PubMed, FDA, WHO, CDC, NIH, TechCrunch, Forbes, MedTech Dive, STAT News, a16z, Bessemer Venture Partners
- **Avoid**: generic blogs, SEO content farms, personal finance sites, social media posts, sites you don't recognize as authoritative
- If Tavily returns a low-quality source, use it but also include the domain of a more authoritative source if you recall the data from it
- **Never include meta-commentary in reasoning** like "my search was unhelpful" or "I couldn't find data" — just score based on what you know and note the limitation in suggestions instead"""


REACT_SYSTEM_PROMPT_INVESTOR = """You are PitchPal, an expert AI investment analyst that uses the ReAct (Reasoning + Acting) pattern.

You will evaluate a startup pitch FROM AN INVESTOR'S PERSPECTIVE — focus on ROI potential, risk, scalability, and exit opportunities.
Be rigorous and data-driven. Investors need hard numbers and realistic assessments.

{tool_descriptions}

## How to respond

You MUST follow this EXACT format for each step. Do NOT deviate:

Thought: <your reasoning about what you need to research next>
Action: <tool_name>("your search query")

After you receive an observation, continue with another Thought/Action pair OR finish with a final evaluation.

When you have gathered enough research (typically 5-6 searches), finish with:

Thought: I now have sufficient research data to provide a comprehensive investment analysis.
Action: final_evaluation(<the complete JSON evaluation>)

## Research Strategy (follow this order)
You MUST research each of these areas with dedicated searches:
1. **Market size & investment thesis** — "[industry] market size 2024 growth rate investment"
2. **Revenue & unit economics benchmarks** — "[SaaS/marketplace/etc] CAC LTV gross margin benchmarks [industry]"
3. **Competitors & recent funding rounds** — "[company type] startups funded 2023 2024 series A crunchbase"
4. **Exit comparables** — "[industry] M&A acquisitions IPO valuations comparable companies"
5. **Team track record** — "[founder background/company/university] healthcare AI startups founders credentials"
6. **Risk factors** — "[regulatory/technical/market] risks [industry] startups"

## Rules
- Always start with a Thought
- Use 5-6 tool calls before evaluating — one search per dimension area
- Be MORE critical than the startup evaluation — investors need realistic risk assessment
- Reasoning must cite ACTUAL data found, not just restate the pitch
- Every dimension must have at least 1 source URL — NEVER leave sources as []
- Keep each reasoning field CONCISE (under 200 characters) to avoid truncation

## CRITICAL: Source Requirements
- Every dimension MUST have at least 1 real URL in the `sources` array
- NEVER leave sources as an empty array [] — always include URLs from your actual research
- Reasoning must reference real findings ("Rock Health: 60% of health AI startups fail FDA clearance") not pitch claims

## Final Evaluation JSON Format
The final_evaluation action MUST contain ONLY a valid JSON object — no extra text before or after:
{{
  "market_opportunity": {{
    "score": 7,
    "reasoning": "Grand View Research: AI diagnostics $4.9B in 2023, 44% CAGR. Large but crowded segment.",
    "suggestions": ["Define SAM (serviceable addressable market) not just TAM"],
    "sources": ["https://grandviewresearch.com/industry-analysis/ai-diagnostics-market"],
    "benchmark": "AI diagnostics TAM: $4.9B (2023), growing to $45B by 2030"
  }},
  "revenue_economics": {{
    "score": 6,
    "reasoning": "Medical SaaS peers: 70-80% gross margin, CAC $15-50K per hospital. LTV:CAC needs to exceed 3:1.",
    "suggestions": ["Show 12-month net revenue retention", "Document exact CAC by channel"],
    "sources": ["https://a16z.com/saas-metrics", "https://openviewpartners.com/saas-benchmarks"],
    "benchmark": "Top-quartile medical SaaS: 80% gross margin, LTV:CAC of 5:1+"
  }},
  "scalability": {{
    "score": 7,
    "reasoning": "Cloud-based models scale well. Aidoc scaled to 1,000+ hospitals without proportional cost increase.",
    "suggestions": ["Document infrastructure costs at 10x current scale"],
    "sources": ["https://techcrunch.com/aidoc-raises"],
    "benchmark": "Comparable SaaS: marginal cost per new customer < 5% of ACV at scale"
  }},
  "competitive_moat": {{
    "score": 5,
    "reasoning": "Aidoc ($110M), Viz.ai ($251M), Qure.ai are well-funded. Differentiation requires strong IP or exclusives.",
    "suggestions": ["File patents", "Pursue FDA 510(k) clearance as a moat"],
    "sources": ["https://crunchbase.com/organization/aidoc", "https://crunchbase.com/organization/viz-ai"],
    "benchmark": "Competitor Aidoc: deployed in 1,200+ hospitals, 10+ FDA clearances"
  }},
  "team_execution": {{
    "score": 7,
    "reasoning": "Stanford ML PhD + hospital CTO combo is strong. Rock Health: 60% of health AI unicorn founders have clinical experience.",
    "suggestions": ["Add regulatory/FDA affairs advisor", "Demonstrate prior startup exits or enterprise sales"],
    "sources": ["https://rockhealth.com/reports", "https://stanford.edu/"],
    "benchmark": "Rock Health: Top health AI teams have 2+ clinical domain experts + 1 technical PhD"
  }},
  "risk_assessment": {{
    "score": 6,
    "reasoning": "FDA clearance takes 18-36 months and costs $1-3M+. Reimbursement from CMS is uncertain for AI diagnostics.",
    "suggestions": ["Budget $2M+ for FDA pathway", "Engage reimbursement consultant now"],
    "sources": ["https://fda.gov/medical-devices/software-medical-device", "https://cms.gov/medicare-coverage"],
    "benchmark": "FDA 510(k) approval: avg 177 days; PMA pathway: avg 3 years"
  }},
  "exit_potential": {{
    "score": 7,
    "reasoning": "Philips acquired BioTelemetry ($2.8B). Siemens, GE Healthcare active acquirers in AI diagnostics space.",
    "suggestions": ["Track acquirer M&A activity", "Build relationships with corporate VC arms of strategic buyers"],
    "sources": ["https://medtechdive.com/philips-biotelemetry", "https://crunchbase.com/acquisitions"],
    "benchmark": "Recent med-AI exits: 4-8x revenue multiples for FDA-cleared products"
  }},
  "overall_score": 6.5,
  "recommendation": "Strong Buy/Buy/Hold/Pass/Strong Pass",
  "strengths": ["data-backed strength", "strength2", "strength3"],
  "concerns": ["evidence-backed concern", "concern2", "concern3"],
  "next_steps": ["step1", "step2", "step3"],
  "contradictions": [
    {{
      "pitch_claim": "exact claim the founder made",
      "research_finding": "what your research actually found that contradicts it",
      "source": "domain.com or full URL"
    }}
  ]
}}

## Trust Signal Rules
- `sources`: REQUIRED for every dimension — include URLs from your actual research. Never use [] empty array.
- `reasoning`: Must reference actual data, not pitch claims. Bad: "The team is experienced." Good: "Rock Health: Only 12% of health AI founders have both ML PhD + clinical ops background."
- `benchmark`: Include real 1-line industry comparisons whenever found. Set null ONLY if truly nothing found.
- `contradictions`: CRITICAL — if any pitch claim is contradicted by research, list it here.

## Source Quality Rules
- **Prefer authoritative sources**: Grand View Research, McKinsey, Deloitte, Rock Health, CB Insights, Crunchbase, PubMed, FDA, WHO, CDC, NIH, TechCrunch, Forbes, MedTech Dive, STAT News, Fierce Healthcare, a16z, Bessemer Venture Partners, Pitchbook
- **Avoid**: generic blogs, SEO content farms, personal finance sites, social media, unrecognized domains
- If Tavily returns a low-quality source, use it but prefer the most authoritative result available
- **Never include meta-commentary in reasoning** like "my search was unhelpful" or "I couldn't find data" — score based on available knowledge and note gaps in suggestions instead"""


PITCH_PROMPT = """## Startup Pitch to Evaluate

**Startup Name:** {startup_name}

**Pitch:**
{pitch_text}

Begin your research and evaluation now. Start with a Thought about what you need to research first."""


# =============================
# JSON Repair Utilities
# =============================

def _try_fix_truncated_json(raw: str) -> Optional[dict]:
    """
    Attempt to repair truncated/malformed JSON using multiple strategies.
    Returns parsed dict on success, None on failure.
    """
    # Strategy 1: direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Strategy 2: strip trailing commas
    try:
        fixed = re.sub(r",\s*([}\]])", r"\1", raw)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Strategy 3: extract largest {...} block
    try:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end > start:
            chunk = raw[start:end + 1]
            chunk = re.sub(r",\s*([}\]])", r"\1", chunk)
            return json.loads(chunk)
    except json.JSONDecodeError:
        pass

    # Strategy 4: try to close truncated JSON by appending closing braces
    try:
        start = raw.find("{")
        if start != -1:
            partial = raw[start:]
            # Count unclosed braces/brackets
            depth_brace = 0
            depth_bracket = 0
            in_str = False
            escape = False
            for ch in partial:
                if escape:
                    escape = False
                    continue
                if ch == '\\' and in_str:
                    escape = True
                    continue
                if ch == '"' and not escape:
                    in_str = not in_str
                    continue
                if in_str:
                    continue
                if ch == '{':
                    depth_brace += 1
                elif ch == '}':
                    depth_brace -= 1
                elif ch == '[':
                    depth_bracket += 1
                elif ch == ']':
                    depth_bracket -= 1

            # Close string if open
            if in_str:
                partial += '"'
            # Close open brackets
            partial += ']' * max(0, depth_bracket)
            # Close open braces
            partial += '}' * max(0, depth_brace)

            partial = re.sub(r",\s*([}\]])", r"\1", partial)
            return json.loads(partial)
    except (json.JSONDecodeError, Exception):
        pass

    # Strategy 5: extract field by field using regex
    try:
        result = {}
        # Extract numeric fields
        for key in ["overall_score"]:
            m = re.search(rf'"{key}"\s*:\s*([\d.]+)', raw)
            if m:
                result[key] = float(m.group(1))
        # Extract string fields
        for key in ["recommendation"]:
            m = re.search(rf'"{key}"\s*:\s*"([^"]+)"', raw)
            if m:
                result[key] = m.group(1)
        if result:
            return result  # Partial — caller will fill defaults
    except Exception:
        pass

    return None


def _extract_json_string_aware(text: str, open_pos: int) -> str:
    """
    Extract content between balanced parentheses, correctly handling
    string literals (so parentheses inside strings don't confuse depth counting).
    """
    if open_pos >= len(text) or text[open_pos] != '(':
        return text[open_pos:]

    depth = 0
    i = open_pos
    in_str = False
    escape = False

    while i < len(text):
        ch = text[i]

        if escape:
            escape = False
            i += 1
            continue

        if ch == '\\' and in_str:
            escape = True
            i += 1
            continue

        if ch == '"' and not escape:
            in_str = not in_str
            i += 1
            continue

        if in_str:
            i += 1
            continue

        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth == 0:
                return text[open_pos + 1: i]

        i += 1

    # If we never closed, return everything after the opening paren
    return text[open_pos + 1:]


class ReActAgent:
    """
    Real ReAct (Reasoning + Acting) agent for startup pitch evaluation.
    Supports startup (5 dims) and investor (7 dims) roles.
    """

    WINDOW_SIZE = 4

    def __init__(self):
        self.llm = LLMClient()
        self.tools = ToolRegistry()
        self.max_steps = settings.MAX_REACT_STEPS

    def _compress_exchange(self, exchange: dict) -> str:
        tool = exchange.get("action", "unknown")
        query = exchange.get("query", "")[:60]
        obs = exchange.get("observation", "")
        first_line = obs.split("\n")[0][:120] if obs else "No result"
        return f"[Prior research] {tool}(\"{query}\") => {first_line}"

    def _build_conversation(self, base_prompt: str, exchanges: list) -> str:
        if len(exchanges) == 0:
            return base_prompt

        conversation = base_prompt

        if len(exchanges) > self.WINDOW_SIZE:
            old_exchanges = exchanges[:-self.WINDOW_SIZE]
            recent_exchanges = exchanges[-self.WINDOW_SIZE:]
            summaries = [self._compress_exchange(ex) for ex in old_exchanges]
            conversation += "\n\n## Research completed so far:\n" + "\n".join(summaries)
            conversation += "\n\n## Recent steps (full detail):"
            for ex in recent_exchanges:
                conversation += f"\n\n{ex['raw_response']}\n\nObservation: {ex['observation']}"
        else:
            for ex in exchanges:
                conversation += f"\n\n{ex['raw_response']}\n\nObservation: {ex['observation']}"

        conversation += "\n\nContinue with your next Thought and Action."
        return conversation

    async def evaluate_pitch(
        self,
        pitch_text: str,
        startup_name: str,
        role: str = "startup",
        on_step: Optional[Callable] = None,
    ) -> tuple[PitchEvaluation, List[AgentStep]]:
        """Run the full ReAct loop. Role determines evaluation dimensions."""

        if role == "investor":
            system_template = REACT_SYSTEM_PROMPT_INVESTOR
        else:
            system_template = REACT_SYSTEM_PROMPT_STARTUP

        system_prompt = system_template.format(
            tool_descriptions=self.tools.get_tool_descriptions()
        )
        pitch_prompt = PITCH_PROMPT.format(
            startup_name=startup_name, pitch_text=pitch_text
        )

        base_prompt = f"{system_prompt}\n\n{pitch_prompt}"
        exchanges: list = []
        steps: List[AgentStep] = []
        step_number = 0

        dim_count = len(INVESTOR_DIMENSIONS if role == "investor" else STARTUP_DIMENSIONS)

        for iteration in range(self.max_steps):
            remaining = self.max_steps - iteration
            conversation = self._build_conversation(base_prompt, exchanges)

            # ── Budget warning: force final_evaluation before we run out ──
            if remaining <= 2 and exchanges:
                dim_keys = INVESTOR_DIMENSIONS if role == "investor" else STARTUP_DIMENSIONS
                dim_list = ", ".join(f'"{k}"' for k in dim_keys)
                conversation += (
                    f"\n\n⚠️ CRITICAL — FINAL STEP: You have {remaining} step(s) left. "
                    f"You MUST call final_evaluation NOW. "
                    f"Do NOT search anymore. "
                    f"Score ALL {dim_count} dimensions ({dim_list}) in your JSON. "
                    f"Every dimension MUST have a real score between 0-10 based on the research above."
                )

            response = await self.llm.generate(conversation)
            thought, action_name, action_input = self._parse_response(response)

            if thought:
                step_number += 1
                thought_step = AgentStep(step_number=step_number, step_type="thought", content=thought)
                steps.append(thought_step)
                if on_step:
                    await on_step(thought_step)

            if action_name == "final_evaluation":
                step_number += 1
                final_step = AgentStep(
                    step_number=step_number,
                    step_type="final_answer",
                    content="Generating structured evaluation...",
                    tool_name="final_evaluation",
                )
                steps.append(final_step)
                if on_step:
                    await on_step(final_step)

                evaluation = await self._parse_evaluation_with_retry(
                    action_input, startup_name, role, pitch_text, exchanges
                )
                return evaluation, steps

            if action_name and action_input:
                step_number += 1
                action_step = AgentStep(
                    step_number=step_number,
                    step_type="action",
                    content=f"Searching: {action_input}",
                    tool_name=action_name,
                    tool_input=action_input,
                )
                steps.append(action_step)
                if on_step:
                    await on_step(action_step)

                observation = await self.tools.execute(action_name, action_input)

                step_number += 1
                obs_step = AgentStep(
                    step_number=step_number,
                    step_type="observation",
                    content=observation[:500],
                    tool_name=action_name,
                )
                steps.append(obs_step)
                if on_step:
                    await on_step(obs_step)

                exchanges.append({
                    "thought": thought or "",
                    "action": action_name,
                    "query": action_input,
                    "observation": observation,
                    "raw_response": response,
                })
            else:
                exchanges.append({
                    "thought": thought or "",
                    "action": "invalid",
                    "query": "",
                    "observation": "You must respond with a Thought followed by an Action.",
                    "raw_response": response,
                })

        return await self._fallback_evaluation(pitch_text, startup_name, role, steps, on_step)

    # ═══════════════════════════════════════════════════════════
    # Parsing
    # ═══════════════════════════════════════════════════════════

    def _clean_json_input(self, raw: str) -> str:
        cleaned = raw.strip()
        if (cleaned.startswith('"') and cleaned.endswith('"')) or \
           (cleaned.startswith("'") and cleaned.endswith("'")):
            cleaned = cleaned[1:-1]
        cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"\n?```\s*$", "", cleaned, flags=re.MULTILINE)
        return cleaned.strip()

    def _parse_response(self, response: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
        thought = None
        action_name = None
        action_input = None

        logger.info(f"=== RAW LLM (first 300) ===\n{response[:300]}")

        thought_match = re.search(r"Thought:\s*(.+?)(?=\nAction:|\Z)", response, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()

        # ── Strategy 1: explicit final_evaluation(...) call ──
        final_eval_match = re.search(r"Action:\s*final_evaluation\s*\(", response)
        if final_eval_match:
            action_name = "final_evaluation"
            paren_pos = response.index("(", final_eval_match.start())
            # Use string-aware extractor (FIX: was blind bracket counter)
            raw_content = _extract_json_string_aware(response, paren_pos)
            action_input = self._clean_json_input(raw_content)
            logger.info(f"Extracted final_evaluation, length={len(action_input)}")
            return thought, action_name, action_input

        # ── Strategy 2: JSON block directly in response ──
        if "{" in response and "overall_score" in response:
            action_name = "final_evaluation"
            action_input = self._clean_json_input(response)
            logger.info("Detected inline JSON final evaluation")
            return thought, action_name, action_input

        # ── Strategy 3: regular tool Action ──
        action_match = re.search(r'Action:\s*(\w+)\s*\(\s*["\']?(.*?)["\']?\s*\)', response, re.DOTALL)
        if action_match:
            action_name = action_match.group(1).strip()
            action_input = action_match.group(2).strip()
            return thought, action_name, action_input

        return thought, action_name, action_input

    async def _parse_evaluation_with_retry(
        self,
        raw_json: str,
        startup_name: str,
        role: str,
        pitch_text: str,
        exchanges: list,
    ) -> PitchEvaluation:
        """
        Parse evaluation JSON. If parsing fails, make a targeted retry LLM call
        to get a clean JSON — never return the dummy 'parsing failed' result.
        """
        logger.info(f"=== PARSING EVALUATION (role={role}), raw len={len(raw_json or '')} ===")

        # Attempt 1: parse as-is
        if raw_json:
            result = self._attempt_parse(raw_json, startup_name, role)
            if result:
                logger.info("Parse attempt 1 succeeded")
                return result

        # Attempt 2: retry with a dedicated clean-JSON prompt
        logger.warning("Parse attempt 1 failed — retrying with clean JSON prompt")
        result = await self._retry_clean_json(startup_name, role, pitch_text, exchanges)
        if result:
            logger.info("Parse retry succeeded")
            return result

        # Attempt 3: absolute fallback (this should almost never be reached)
        logger.error("All parse attempts failed — using structural fallback")
        return self._create_structural_fallback(startup_name, role)

    def _attempt_parse(self, raw_json: str, startup_name: str, role: str) -> Optional[PitchEvaluation]:
        """Try to parse raw_json into a PitchEvaluation. Returns None on failure."""
        cleaned = self._clean_json_input(raw_json)

        data = _try_fix_truncated_json(cleaned)
        if data is None:
            # One more try: from the raw string
            data = _try_fix_truncated_json(raw_json)

        if data is None:
            return None

        return self._build_evaluation_from_dict(data, startup_name, role)

    async def _retry_clean_json(
        self, startup_name: str, role: str, pitch_text: str, exchanges: list
    ) -> Optional[PitchEvaluation]:
        """
        Ask the LLM directly: 'Return ONLY valid JSON scores for this pitch.'
        This is used when the primary evaluation JSON failed to parse.
        """
        if role == "investor":
            dim_keys = INVESTOR_DIMENSIONS
            fields_example = (
                '"market_opportunity": {"score": 7, "reasoning": "...", "suggestions": ["s1"]}, '
                '"revenue_economics": {"score": 7, "reasoning": "...", "suggestions": ["s1"]}, '
                '"scalability": {"score": 7, "reasoning": "...", "suggestions": ["s1"]}, '
                '"competitive_moat": {"score": 7, "reasoning": "...", "suggestions": ["s1"]}, '
                '"team_execution": {"score": 7, "reasoning": "...", "suggestions": ["s1"]}, '
                '"risk_assessment": {"score": 7, "reasoning": "...", "suggestions": ["s1"]}, '
                '"exit_potential": {"score": 7, "reasoning": "...", "suggestions": ["s1"]}'
            )
        else:
            dim_keys = STARTUP_DIMENSIONS
            fields_example = (
                '"problem_clarity": {"score": 7, "reasoning": "...", "suggestions": ["s1"]}, '
                '"market_opportunity": {"score": 7, "reasoning": "...", "suggestions": ["s1"]}, '
                '"business_model": {"score": 7, "reasoning": "...", "suggestions": ["s1"]}, '
                '"competitive_advantage": {"score": 7, "reasoning": "...", "suggestions": ["s1"]}, '
                '"team_strength": {"score": 7, "reasoning": "...", "suggestions": ["s1"]}'
            )

        research_summary = "\n".join(
            f"- {ex.get('action','tool')}({ex.get('query','')[:50]}): {ex.get('observation','')[:200]}"
            for ex in exchanges[-6:]  # last 6 exchanges for context
        )

        retry_prompt = f"""You are a startup evaluator. Based on the research below, return ONLY a valid JSON object — no intro text, no explanation, JUST the JSON.

Startup: {startup_name}
Pitch summary: {pitch_text[:400]}

Research gathered:
{research_summary}

Return ONLY this JSON (fill in real scores 0-10 and concise reasoning):
{{
  {fields_example},
  "overall_score": 7,
  "recommendation": "Buy",
  "strengths": ["strength1", "strength2", "strength3"],
  "concerns": ["concern1", "concern2"],
  "next_steps": ["step1", "step2"]
}}

IMPORTANT: Return ONLY the JSON. No text before or after. No markdown code blocks."""

        try:
            response = await self.llm.generate(retry_prompt)
            logger.info(f"Retry response (first 200): {response[:200]}")
            return self._attempt_parse(response, startup_name, role)
        except Exception as e:
            logger.error(f"Retry LLM call failed: {e}")
            return None

    def _build_evaluation_from_dict(self, data: dict, startup_name: str, role: str) -> PitchEvaluation:
        """Build a PitchEvaluation from a successfully parsed dict."""
        logger.info(f"Building evaluation from dict. Keys: {list(data.keys())}")

        if role == "investor":
            dim_keys = INVESTOR_DIMENSIONS
            dim_titles = INVESTOR_DIMENSION_TITLES
        else:
            dim_keys = STARTUP_DIMENSIONS
            dim_titles = STARTUP_DIMENSION_TITLES

        dimensions = []
        for key in dim_keys:
            d = data.get(key, {})
            if not isinstance(d, dict):
                d = {}
            dimensions.append(DimensionScore(
                name=dim_titles.get(key, key.replace("_", " ").title()),
                score=float(d.get("score", 5.0)),
                reasoning=d.get("reasoning", "Analysis based on available data"),
                suggestions=d.get("suggestions", ["Further research recommended"]),
                sources=d.get("sources", []),
                benchmark=d.get("benchmark") or None,
            ))

        # Derive overall_score if missing
        if "overall_score" not in data or data["overall_score"] is None:
            scores = [dim.score for dim in dimensions if dim.score > 0]
            data["overall_score"] = round(sum(scores) / len(scores), 1) if scores else 5.0

        overall = float(data.get("overall_score", 5.0))

        # Derive recommendation if missing
        rec = data.get("recommendation") or data.get("investment_recommendation")
        if not rec:
            rec = (
                "Strong Buy" if overall >= 8.5
                else "Buy" if overall >= 7
                else "Hold" if overall >= 5
                else "Pass" if overall >= 3
                else "Strong Pass"
            )

        # Extract contradictions
        raw_contradictions = data.get("contradictions", [])
        contradictions = []
        for c in raw_contradictions:
            if isinstance(c, dict) and c.get("pitch_claim") and c.get("research_finding"):
                contradictions.append(Contradiction(
                    pitch_claim=c["pitch_claim"],
                    research_finding=c["research_finding"],
                    source=c.get("source", ""),
                ))

        return PitchEvaluation(
            startup_name=startup_name,
            overall_score=overall,
            investment_recommendation=rec,
            role=role,
            dimensions=dimensions,
            key_strengths=data.get("strengths", ["Value proposition identified"]),
            main_concerns=data.get("concerns", ["Further validation needed"]),
            next_steps=data.get("next_steps", ["Conduct deeper due diligence"]),
            contradictions=contradictions,
        )

    # ═══════════════════════════════════════════════════════════
    # Fallbacks
    # ═══════════════════════════════════════════════════════════

    async def _fallback_evaluation(self, pitch_text, startup_name, role, steps, on_step):
        """Called when agent hits MAX_REACT_STEPS without a final_evaluation."""
        step_number = len(steps) + 1
        fallback_step = AgentStep(
            step_number=step_number,
            step_type="thought",
            content="Max steps reached. Generating evaluation from gathered research...",
        )
        steps.append(fallback_step)
        if on_step:
            await on_step(fallback_step)

        # Use exchanges from steps
        observations = "\n".join(
            f"- {s.content}" for s in steps if s.step_type == "observation"
        )

        dim_keys = INVESTOR_DIMENSIONS if role == "investor" else STARTUP_DIMENSIONS
        dim_fields = ", ".join([
            f'"{k}": {{"score": 7, "reasoning": "...", "suggestions": ["s1"]}}'
            for k in dim_keys
        ])

        prompt = f"""You are an expert pitch evaluator. Based on the research below, score this startup.
Startup: {startup_name}
Pitch: {pitch_text[:500]}
Research: {observations[:1000]}

Return ONLY valid JSON (no text before or after):
{{ {dim_fields}, "overall_score": 7, "recommendation": "Buy", "strengths": ["a","b","c"], "concerns": ["a","b"], "next_steps": ["a","b"] }}"""

        try:
            response = await self.llm.generate(prompt)
            evaluation = self._attempt_parse(response, startup_name, role)
            if evaluation is None:
                evaluation = self._create_structural_fallback(startup_name, role)
        except Exception as e:
            logger.error(f"Fallback LLM call failed: {e}")
            evaluation = self._create_structural_fallback(startup_name, role)

        step_number += 1
        final_step = AgentStep(
            step_number=step_number,
            step_type="final_answer",
            content="Evaluation generated from research.",
            tool_name="final_evaluation",
        )
        steps.append(final_step)
        if on_step:
            await on_step(final_step)

        return evaluation, steps

    def _create_structural_fallback(self, startup_name: str, role: str = "startup") -> PitchEvaluation:
        """
        Last-resort fallback. Returns a neutral evaluation with a clear note
        that it requires re-evaluation. Should almost never be reached after
        the retry logic above.
        """
        dim_keys = INVESTOR_DIMENSIONS if role == "investor" else STARTUP_DIMENSIONS
        dim_titles = INVESTOR_DIMENSION_TITLES if role == "investor" else STARTUP_DIMENSION_TITLES

        dimensions = [
            DimensionScore(
                name=dim_titles.get(k, k.replace("_", " ").title()),
                score=5.0,
                reasoning="Unable to generate evaluation — please re-submit the pitch",
                suggestions=["Re-run the evaluation for accurate scores"],
            )
            for k in dim_keys
        ]

        return PitchEvaluation(
            startup_name=startup_name,
            overall_score=5.0,
            investment_recommendation="Hold — Re-evaluation Required",
            role=role,
            dimensions=dimensions,
            key_strengths=["Please re-submit pitch for full evaluation"],
            main_concerns=["Evaluation could not be completed — network or model issue"],
            next_steps=["Re-submit pitch", "Check backend logs for details"],
        )

    # ── Legacy wrapper kept for backward compat ──────────────
    def _parse_evaluation(self, raw_json: str, startup_name: str, role: str = "startup") -> PitchEvaluation:
        """Sync wrapper — use _parse_evaluation_with_retry for new code."""
        if not raw_json:
            return self._create_structural_fallback(startup_name, role)
        result = self._attempt_parse(raw_json, startup_name, role)
        return result if result else self._create_structural_fallback(startup_name, role)
