"""
PitchPal v2 - Real Tools for ReAct Agent
Uses Tavily Search API for actual web research.
Includes semantic caching — skips Tavily for similar queries, saves 40-60% tokens.
"""

import re
from typing import Dict, Any
from tavily import TavilyClient
from app.config import settings
from app.agent.semantic_cache import semantic_cache


# ── Quality English-language sources for startup/VC research ─────────────────
PREFERRED_DOMAINS = [
    "techcrunch.com", "crunchbase.com", "pitchbook.com", "forbes.com",
    "businessinsider.com", "bloomberg.com", "reuters.com", "wsj.com",
    "ft.com", "economist.com", "mckinsey.com", "bcg.com", "hbr.org",
    "statista.com", "grandviewresearch.com", "mordorintelligence.com",
    "marketsandmarkets.com", "ibisworld.com", "cb-insights.com",
    "a16z.com", "sequoiacap.com", "ycombinator.com", "venturebeat.com",
    "tracxn.com", "dealroom.co", "sifted.eu", "axios.com",
]

# ── Domains known to return irrelevant/dictionary/non-English junk ───────────
EXCLUDED_DOMAINS = [
    "dictionary.cambridge.org", "merriam-webster.com", "en.wikipedia.org",
    "zh.wikipedia.org", "ja.wikipedia.org", "ko.wikipedia.org",
    "baidu.com", "sina.com", "weibo.com", "163.com",
]

# Unicode CJK range — used to detect non-English content
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]")


def _clean_text(raw: str) -> str:
    """
    Strip markdown syntax and whitespace noise from Tavily results
    before returning them to the agent.
    """
    if not raw:
        return raw
    # Remove markdown headings
    text = re.sub(r"#{1,6}\s*", "", raw)
    # Remove bold/italic markers
    text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)
    # Remove markdown links, keep link text
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    # Collapse multiple newlines/spaces
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _is_english_content(text: str) -> bool:
    """Return False if text contains significant CJK (Chinese/Japanese/Korean) characters."""
    cjk_chars = len(_CJK_RE.findall(text))
    total_chars = len(text.replace(" ", ""))
    if total_chars == 0:
        return True
    return (cjk_chars / total_chars) < 0.05  # less than 5% CJK = OK


def _format_result(response: dict, fallback_msg: str) -> str:
    """
    Format a Tavily response into a clean text summary.
    Skips non-English content and strips markdown.
    Keeps source domain + URL so the LLM can cite them in evaluations.
    """
    results = []

    # Include AI answer summary if present
    if response.get("answer"):
        answer = _clean_text(response["answer"])
        if _is_english_content(answer):
            results.append(f"Summary: {answer}")

    # Include top results, filtering out junk
    for r in response.get("results", [])[:3]:
        title   = _clean_text(r.get("title", ""))
        content = _clean_text(r.get("content", ""))
        url     = r.get("url", "")

        # Skip non-English results
        if not _is_english_content(title) or not _is_english_content(content):
            continue

        # Skip dictionary/encyclopaedia noise
        if any(skip in url for skip in EXCLUDED_DOMAINS):
            continue

        # Extract readable domain for attribution (e.g. "grandviewresearch.com")
        try:
            domain = url.split("//")[-1].split("/")[0].replace("www.", "")
        except Exception:
            domain = ""

        source_tag = f" [{domain}]" if domain else ""
        url_line   = f"  Source: {url}" if url else ""

        if content:
            snippet = content[:200].rstrip()
            results.append(f"- {title}{source_tag}: {snippet}{url_line}")
        elif title:
            results.append(f"- {title}{source_tag}{url_line}")

    return "\n".join(results) if results else fallback_msg



class ToolRegistry:
    """Registry of all available tools for the ReAct agent."""

    def __init__(self):
        self.tavily = TavilyClient(api_key=settings.TAVILY_API_KEY)
        self.tools = {
            "market_research": self.market_research,
            "competitor_analysis": self.competitor_analysis,
            "financial_analysis": self.financial_analysis,
            "industry_trends": self.industry_trends,
        }

    def get_tool_descriptions(self) -> str:
        """Return formatted tool descriptions for the agent prompt."""
        return """Available Tools:

1. market_research(query)
   - Search for real market data: market size (TAM/SAM/SOM), growth rates, industry reports
   - Example: market_research("healthtech AI diagnostics market size 2024")

2. competitor_analysis(query)
   - Find real competitors, their funding, market position, and differentiators
   - Example: competitor_analysis("AI medical imaging startups competitors funding")

3. financial_analysis(query)
   - Research business model viability, unit economics benchmarks, pricing strategies
   - Example: financial_analysis("SaaS healthcare B2B pricing benchmarks LTV CAC")

4. industry_trends(query)
   - Search for latest industry trends, regulatory changes, technology shifts
   - Example: industry_trends("AI healthcare regulation FDA approval trends 2024")"""

    async def execute(self, tool_name: str, tool_input: str) -> str:
        """Execute a tool by name and return results."""
        if tool_name not in self.tools:
            return f"Error: Unknown tool '{tool_name}'. Available tools: {list(self.tools.keys())}"

        try:
            result = await self.tools[tool_name](tool_input)
            return result
        except Exception as e:
            return f"Tool execution error: {str(e)}"

    def _tavily_search(self, search_query: str, fallback_msg: str) -> str:
        """
        Shared Tavily call with domain filtering and content cleaning.
        No caching here — callers handle cache logic.
        """
        response = self.tavily.search(
            query=search_query,
            search_depth="basic",
            max_results=3,
            include_answer=True,
            exclude_domains=EXCLUDED_DOMAINS,
        )
        return _format_result(response, fallback_msg)

    async def market_research(self, query: str) -> str:
        """Search for market data using Tavily (with semantic cache)."""
        search_query = f"{query} market size TAM growth rate industry report 2024 2025"

        cached = semantic_cache.get("market_research", search_query)
        if cached:
            return cached

        try:
            result = self._tavily_search(
                search_query,
                "No market data found for this query."
            )
            semantic_cache.set("market_research", search_query, result)
            return result
        except Exception as e:
            return f"Market research search failed: {str(e)}"

    async def competitor_analysis(self, query: str) -> str:
        """Search for competitor information using Tavily (with semantic cache)."""
        search_query = f"{query} competitors startups companies funding raised 2024"

        cached = semantic_cache.get("competitor_analysis", search_query)
        if cached:
            return cached

        try:
            result = self._tavily_search(
                search_query,
                "No competitor data found."
            )
            semantic_cache.set("competitor_analysis", search_query, result)
            return result
        except Exception as e:
            return f"Competitor analysis search failed: {str(e)}"

    async def financial_analysis(self, query: str) -> str:
        """Search for financial/business model benchmarks using Tavily (with semantic cache)."""
        search_query = f"{query} unit economics gross margin LTV CAC benchmark startup"

        cached = semantic_cache.get("financial_analysis", search_query)
        if cached:
            return cached

        try:
            result = self._tavily_search(
                search_query,
                "No financial benchmarks found."
            )
            semantic_cache.set("financial_analysis", search_query, result)
            return result
        except Exception as e:
            return f"Financial analysis search failed: {str(e)}"

    async def industry_trends(self, query: str) -> str:
        """Search for industry trends using Tavily (with semantic cache)."""
        search_query = f"{query} industry trends outlook 2024 2025 venture capital investment"

        cached = semantic_cache.get("industry_trends", search_query)
        if cached:
            return cached

        try:
            result = self._tavily_search(
                search_query,
                "No trend data found."
            )
            semantic_cache.set("industry_trends", search_query, result)
            return result
        except Exception as e:
            return f"Industry trends search failed: {str(e)}"
