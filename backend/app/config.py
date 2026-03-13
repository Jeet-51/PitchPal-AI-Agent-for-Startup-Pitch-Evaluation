"""
PitchPal v2 - Configuration
Environment variables and app settings
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Search for .env in current dir, backend/, or parent backend/ from app/
env_paths = [
    Path(".env"),
    Path("backend/.env"),
    Path(__file__).parent.parent / ".env",
]

for env_path in env_paths:
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        break
else:
    load_dotenv()


class Settings:
    # LLM Configuration
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini")  # groq | gemini | openai | anthropic
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

    # Model names per provider
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

    # Tavily Search
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")

    # App Settings
    MAX_REACT_STEPS: int = int(os.getenv("MAX_REACT_STEPS", "12"))
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))

    # Investor Access
    INVESTOR_ACCESS_CODE: str = os.getenv("INVESTOR_ACCESS_CODE", "")

    # Rate Limiting
    RATE_LIMIT_STARTUP_MAX: int = int(os.getenv("RATE_LIMIT_STARTUP_MAX", "3"))
    RATE_LIMIT_INVESTOR_MAX: int = int(os.getenv("RATE_LIMIT_INVESTOR_MAX", "5"))
    RATE_LIMIT_WINDOW_HOURS: int = int(os.getenv("RATE_LIMIT_WINDOW_HOURS", "4"))

    # CORS
    FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://localhost:3000")


settings = Settings()

if not settings.INVESTOR_ACCESS_CODE:
    import warnings
    warnings.warn(
        "INVESTOR_ACCESS_CODE is not set — investor role will be inaccessible. "
        "Set it in your .env file.",
        stacklevel=2,
    )
