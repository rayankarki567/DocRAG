from __future__ import annotations

import json
from datetime import date, time

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import crud
from app.services.llm import LLMClient


BOOKING_SYSTEM_PROMPT = """
You are an interview booking extractor.

Determine whether the user wants to book an interview and extract any
booking information present in their message.

Return ONLY valid JSON:

{
  "wants_to_book": true,
  "name": null,
  "email": null,
  "date": null,
  "time": null
}

Rules:
- wants_to_book = true when the user wants to schedule/book an interview
  OR is providing information for an ongoing booking.
- Extract only information explicitly provided by the user.
- name = person's name.
- email = email address.
- date = YYYY-MM-DD.
- time = HH:MM in 24-hour format.
- Use null when a value is missing.
- Never invent values.
"""


class BookingService:
    def __init__(self, redis: Redis, llm: LLMClient) -> None:
        self.redis = redis
        self.llm = llm

    def _key(self, session_id: str) -> str:
        return f"booking:{session_id}"

    async def _get_state(self, session_id: str) -> dict[str, str | None]:
        data = await self.redis.get(self._key(session_id))

        if not data:
            return {
                "name": None,
                "email": None,
                "date": None,
                "time": None,
            }

        return json.loads(data)

    async def _save_state(
        self,
        session_id: str,
        state: dict[str, str | None],
    ) -> None:
        await self.redis.set(
            self._key(session_id),
            json.dumps(state),
            ex=6 * 60 * 60,
        )

    async def handle(
        self,
        db: AsyncSession,
        session_id: str,
        message: str,
    ) -> str | None:

        state = await self._get_state(session_id)

        extracted = await self.llm.extract_json(
            system_prompt=BOOKING_SYSTEM_PROMPT,
            user_prompt=(
                f"Today's date is {date.today().isoformat()}.\n\n"
                f"User message: {message}"
            ),
        )

        wants_to_book = extracted.get("wants_to_book", False)

        # Ignore this message if no booking is happening.
        # Also allow an existing partial booking to continue.
        has_existing_booking = any(state.values())

        if not wants_to_book and not has_existing_booking:
            return None

        # Merge newly extracted values with existing values.
        for field in ("name", "email", "date", "time"):
            if extracted.get(field):
                state[field] = extracted[field]

        await self._save_state(session_id, state)

        missing = [
            field
            for field in ("name", "email", "date", "time")
            if not state[field]
        ]

        if missing:
            prompts = {
                "name": "your full name",
                "email": "your email address",
                "date": "your preferred interview date",
                "time": "your preferred interview time",
            }

            if len(missing) == 1:
                return f"Could you provide {prompts[missing[0]]}?"

            items = [prompts[field] for field in missing]

            return (
                "To book the interview, please provide "
                + ", ".join(items[:-1])
                + f", and {items[-1]}."
            )

        # Validate date/time before inserting into SQL.
        interview_date = date.fromisoformat(state["date"])
        interview_time = time.fromisoformat(state["time"])

        booking = await crud.create_booking(
            db,
            session_id=session_id,
            name=state["name"],
            email=state["email"],
            interview_date=interview_date,
            interview_time=interview_time,
        )

        await self.redis.delete(self._key(session_id))

        return (
            f"Your interview has been booked for "
            f"{booking.interview_date.isoformat()} at "
            f"{booking.interview_time.strftime('%H:%M')}."
        )