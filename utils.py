"""
Helpers: chunking long text (by words), rate limiting.
"""
import time
from collections import defaultdict

from config import MAX_MESSAGE_LENGTH


def split_into_chunks(text: str, max_length: int = MAX_MESSAGE_LENGTH) -> list[str]:
    """
    Split text into chunks not exceeding max_length.
    Tries to break at word boundaries to avoid cutting words/emojis.
    """
    if not text:
        return []
    if len(text) <= max_length:
        return [text]
    chunks = []
    current = []
    current_len = 0
    # Split by whitespace but keep newlines as separators
    for part in text.split():
        part_len = len(part) + (1 if current else 0)  # +1 for space
        if current_len + part_len <= max_length:
            current.append(part)
            current_len += part_len
        else:
            if current:
                chunks.append(" ".join(current))
            if len(part) > max_length:
                # Single token too long: split by chars
                for i in range(0, len(part), max_length):
                    chunks.append(part[i : i + max_length])
                current = []
                current_len = 0
            else:
                current = [part]
                current_len = len(part) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks


class RateLimiter:
    """Simple per-user rate limit (in-memory)."""

    def __init__(self, max_per_minute: int):
        self.max_per_minute = max_per_minute
        self._timestamps: dict[int, list[float]] = defaultdict(list)

    def is_allowed(self, user_id: int) -> bool:
        now = time.monotonic()
        cutoff = now - 60
        self._timestamps[user_id] = [t for t in self._timestamps[user_id] if t > cutoff]
        if len(self._timestamps[user_id]) >= self.max_per_minute:
            return False
        self._timestamps[user_id].append(now)
        return True
