from __future__ import annotations

import json

from redis.asyncio import Redis

from app.config import Settings


class RedisChatMemory:
    """Store and retrieve conversation history using Redis."""

    def __init__(self, redis: Redis, settings: Settings) -> None:
        self.redis = redis
        self.max_history = settings.redis_max_history_messages
        self.ttl = settings.redis_chat_ttl_seconds

    def _key(self, session_id: str) -> str:
        return f"chat:{session_id}"

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
    ) -> None:
        """Store one chat message."""
        key = self._key(session_id)

        message = {
            "role": role,
            "content": content,
        }

        await self.redis.rpush(key, json.dumps(message))

        # Keep only the most recent messages.
        await self.redis.ltrim(
            key,
            -self.max_history,
            -1,
        )

        # Expire inactive conversations automatically.
        await self.redis.expire(key, self.ttl)

    async def get_history(
        self,
        session_id: str,
    ) -> list[dict[str, str]]:
        """Return conversation history in LLM message format."""
        messages = await self.redis.lrange(
            self._key(session_id),
            0,
            -1,
        )

        return [
            json.loads(message)
            for message in messages
        ]

    async def clear(self, session_id: str) -> None:
        """Delete a session's conversation history."""
        await self.redis.delete(self._key(session_id))