"""
PitchPal v2 - LLM Abstraction Layer
Swap between Groq, Gemini, OpenAI, and Anthropic with one env var change.
"""

import json
from typing import AsyncGenerator
from app.config import settings


class LLMClient:
    """
    Unified LLM client that abstracts away provider differences.
    Supports: Groq (free/fast), Gemini, OpenAI, Anthropic.
    """

    def __init__(self):
        self.provider = settings.LLM_PROVIDER
        self._client = None
        self._setup_client()

    def _setup_client(self):
        if self.provider == "groq":
            from groq import AsyncGroq

            self._client = AsyncGroq(api_key=settings.GROQ_API_KEY)

        elif self.provider == "gemini":
            import google.generativeai as genai

            genai.configure(api_key=settings.GEMINI_API_KEY)
            self._client = genai.GenerativeModel(
                model_name=settings.GEMINI_MODEL,
                generation_config={
                    "temperature": settings.LLM_TEMPERATURE,
                    "top_p": 1.0 if settings.LLM_TEMPERATURE == 0 else 0.95,
                    "max_output_tokens": 4096,
                },
            )

        elif self.provider == "openai":
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

        elif self.provider == "anthropic":
            from anthropic import AsyncAnthropic

            self._client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)

        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    async def generate(self, prompt: str) -> str:
        """Generate a response from the LLM."""
        if self.provider == "groq":
            return await self._groq_generate(prompt)
        elif self.provider == "gemini":
            return await self._gemini_generate(prompt)
        elif self.provider == "openai":
            return await self._openai_generate(prompt)
        elif self.provider == "anthropic":
            return await self._anthropic_generate(prompt)

    async def _groq_generate(self, prompt: str) -> str:
        """Generate using Groq (free, very fast)."""
        response = await self._client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=4096,
        )
        return response.choices[0].message.content

    async def _gemini_generate(self, prompt: str) -> str:
        """Generate using Google Gemini."""
        response = await self._client.generate_content_async(prompt)
        return response.text

    async def _openai_generate(self, prompt: str) -> str:
        """Generate using OpenAI."""
        response = await self._client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=4096,
        )
        return response.choices[0].message.content

    async def _anthropic_generate(self, prompt: str) -> str:
        """Generate using Anthropic Claude."""
        response = await self._client.messages.create(
            model=settings.ANTHROPIC_MODEL,
            max_tokens=4096,
            temperature=settings.LLM_TEMPERATURE,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def get_provider_name(self) -> str:
        """Return human-readable provider name."""
        names = {
            "groq": f"Groq {settings.GROQ_MODEL}",
            "gemini": f"Google {settings.GEMINI_MODEL}",
            "openai": f"OpenAI {settings.OPENAI_MODEL}",
            "anthropic": f"Anthropic {settings.ANTHROPIC_MODEL}",
        }
        return names.get(self.provider, self.provider)
