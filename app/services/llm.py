from __future__ import annotations

import json
from typing import Any

from openai import AsyncOpenAI, OpenAIError

from app.config import Settings


class LLMClient:
    """Simple client for an OpenAI-compatible LLM API."""

    def __init__(self, settings: Settings) -> None:
        self.client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
        self.model = settings.llm_model

    async def generate(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 700,
    ) -> str:
        """Generate a normal text response."""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    *messages,
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except OpenAIError as exc:
            raise RuntimeError(f"LLM request failed: {exc}") from exc

        return (response.choices[0].message.content or "").strip()

    async def extract_json(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any]:
        """Extract structured JSON from a user message."""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
        except OpenAIError as exc:
            raise RuntimeError(f"LLM JSON request failed: {exc}") from exc

        content = response.choices[0].message.content or "{}"

        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError("LLM returned invalid JSON.") from exc