"""
PitchPal v2 - Input Security

Three layers of protection:
  1. HTML / script stripping       — prevents XSS via pitch text
  2. Length enforcement            — prevents prompt-stuffing / cost abuse
  3. Prompt injection detection    — catches attempts to hijack the agent
"""

import re
import unicodedata
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Length limits ─────────────────────────────────────────────
MIN_PITCH_LENGTH = 50
MAX_PITCH_LENGTH = 5000
MAX_NAME_LENGTH = 200

# ── Prompt injection signatures ───────────────────────────────
# Covers the most common jailbreak / hijack patterns
_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions?",
    r"disregard\s+(all\s+)?(prior|previous|above)\s+",
    r"forget\s+(all\s+)?(previous|prior|your)\s+",
    r"you\s+are\s+now\s+",
    r"new\s+(system\s+)?instructions?\s*:",
    r"<\s*system\s*>",
    r"\[INST\]",
    r"\[SYSTEM\]",
    r"###\s*instruction",
    r"act\s+as\s+(if\s+(you\s+are|you're)\s+)?",
    r"pretend\s+(you\s+are|to\s+be)\s+",
    r"(jailbreak|dan\s+mode|developer\s+mode|god\s+mode)",
    r"override\s+(all\s+)?(safety|restriction|rule|instruction)",
    r"(print|reveal|show|expose|output)\s+(your\s+)?(system\s+)?prompt",
    r"(print|reveal|show|expose|output)\s+(your\s+)?instructions?",
    r"prompt\s+injection",
    r"you\s+must\s+(now\s+)?respond\s+(only\s+)?as",
]

_INJECTION_RE = re.compile(
    "|".join(_INJECTION_PATTERNS),
    re.IGNORECASE | re.DOTALL,
)


# ── HTML stripping ─────────────────────────────────────────────

def strip_html(text: str) -> str:
    """Remove HTML/script tags and decode common HTML entities."""
    if not text:
        return text

    # Nuke entire script/style blocks (content + tags)
    text = re.sub(
        r"<(script|style)[^>]*>.*?</(script|style)>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    # Remove remaining HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    # Decode common entities
    replacements = {
        "&amp;": "&",
        "&lt;": "<",
        "&gt;": ">",
        "&quot;": '"',
        "&#39;": "'",
        "&apos;": "'",
        "&nbsp;": " ",
    }
    for entity, char in replacements.items():
        text = text.replace(entity, char)

    # Collapse excessive whitespace (3+ newlines → 2)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{3,}", "  ", text)

    return text.strip()


# ── Prompt injection detection ─────────────────────────────────

def detect_prompt_injection(text: str) -> Optional[str]:
    """
    Scan text for prompt injection patterns.
    Returns the matched substring if found, None if clean.
    """
    match = _INJECTION_RE.search(text)
    if match:
        logger.warning(
            f"Prompt injection detected — pattern: '{match.group()[:80]}'"
        )
        return match.group()
    return None


# ── Full sanitization pipeline ─────────────────────────────────

def sanitize_inputs(
    pitch_text: str,
    startup_name: str,
) -> tuple[str, str, Optional[str]]:
    """
    Sanitize and validate pitch text + startup name.

    Returns:
        (clean_pitch, clean_name, error_message)
        error_message is None when everything is valid.
    """
    # Step 0: Unicode normalization (prevents bypass with look-alike chars)
    pitch_text = unicodedata.normalize("NFKD", pitch_text)
    startup_name = unicodedata.normalize("NFKD", startup_name)

    # Step 1: strip HTML from both fields
    clean_pitch = strip_html(pitch_text)
    clean_name = strip_html(startup_name)

    # Step 2: length checks
    if len(clean_pitch) < MIN_PITCH_LENGTH:
        return clean_pitch, clean_name, (
            f"Pitch is too short. Please provide at least {MIN_PITCH_LENGTH} characters."
        )

    if len(clean_pitch) > MAX_PITCH_LENGTH:
        return clean_pitch, clean_name, (
            f"Pitch is too long ({len(clean_pitch):,} chars). "
            f"Maximum allowed is {MAX_PITCH_LENGTH:,} characters."
        )

    if len(clean_name) > MAX_NAME_LENGTH:
        return clean_pitch, clean_name, (
            f"Startup name is too long. Maximum is {MAX_NAME_LENGTH} characters."
        )

    if not clean_name.strip():
        return clean_pitch, clean_name, "Startup name cannot be empty."

    # Step 3: prompt injection checks
    injection_in_pitch = detect_prompt_injection(clean_pitch)
    if injection_in_pitch:
        return clean_pitch, clean_name, (
            "Invalid pitch content detected. Please submit a genuine startup pitch description."
        )

    injection_in_name = detect_prompt_injection(clean_name)
    if injection_in_name:
        return clean_pitch, clean_name, (
            "Invalid startup name detected. Please use the actual name of your startup."
        )

    return clean_pitch, clean_name, None
