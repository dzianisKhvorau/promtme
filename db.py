"""
SQLite storage for user balance, free trial, history, and last prompt.
Async via aiosqlite.
"""
import aiosqlite
from pathlib import Path

import config


def _db_path() -> str:
    path = Path(config.DATABASE_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


async def init_db() -> None:
    """Create tables if they don't exist."""
    async with aiosqlite.connect(_db_path()) as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY,
                free_used INTEGER NOT NULL DEFAULT 0,
                balance INTEGER NOT NULL DEFAULT 0,
                last_prompt TEXT,
                last_category TEXT,
                updated_at TEXT
            )
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                category TEXT NOT NULL,
                prompt_preview TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        """)
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_history_user_id ON history(user_id)"
        )
        await conn.commit()


async def get_user(user_id: int) -> dict:
    """Get or create user. Returns dict: free_used, balance, last_prompt, last_category."""
    async with aiosqlite.connect(_db_path()) as conn:
        conn.row_factory = aiosqlite.Row
        async with conn.execute(
            "SELECT free_used, balance, last_prompt, last_category FROM users WHERE user_id = ?",
            (user_id,),
        ) as cur:
            row = await cur.fetchone()
        if row is not None:
            return {
                "free_used": row["free_used"],
                "balance": row["balance"],
                "last_prompt": row["last_prompt"],
                "last_category": row["last_category"],
            }
        await conn.execute(
            "INSERT INTO users (user_id, free_used, balance, updated_at) VALUES (?, 0, 0, datetime('now'))",
            (user_id,),
        )
        await conn.commit()
    return {"free_used": 0, "balance": 0, "last_prompt": None, "last_category": None}


async def consume_generation(user_id: int) -> bool:
    """
    Use one generation: first FREE_TRIAL_GENERATIONS free, then from balance.
    Returns True if consumed, False if no quota.
    """
    user = await get_user(user_id)
    free_used = user["free_used"]
    balance = user["balance"]
    if free_used < config.FREE_TRIAL_GENERATIONS:
        async with aiosqlite.connect(_db_path()) as conn:
            await conn.execute(
                "UPDATE users SET free_used = free_used + 1, updated_at = datetime('now') WHERE user_id = ?",
                (user_id,),
            )
            await conn.commit()
        return True
    if balance > 0:
        async with aiosqlite.connect(_db_path()) as conn:
            await conn.execute(
                "UPDATE users SET balance = balance - 1, updated_at = datetime('now') WHERE user_id = ?",
                (user_id,),
            )
            await conn.commit()
        return True
    return False


async def add_balance(user_id: int, amount: int) -> int:
    """Add to user balance. Returns new balance."""
    await get_user(user_id)
    async with aiosqlite.connect(_db_path()) as conn:
        await conn.execute(
            "UPDATE users SET balance = balance + ?, updated_at = datetime('now') WHERE user_id = ?",
            (amount, user_id),
        )
        await conn.commit()
        async with conn.execute(
            "SELECT balance FROM users WHERE user_id = ?", (user_id,)
        ) as cur:
            row = await cur.fetchone()
    return row[0] if row else 0


async def update_after_generation(
    user_id: int, category: str, prompt: str, append_history: bool = True
) -> None:
    """Update last_prompt, last_category and optionally append to history."""
    preview = prompt[:200] + ("…" if len(prompt) > 200 else "")
    async with aiosqlite.connect(_db_path()) as conn:
        await conn.execute(
            """UPDATE users SET last_prompt = ?, last_category = ?, updated_at = datetime('now')
               WHERE user_id = ?""",
            (prompt, category, user_id),
        )
        if append_history:
            await conn.execute(
                "INSERT INTO history (user_id, category, prompt_preview) VALUES (?, ?, ?)",
                (user_id, category, preview),
            )
            # Keep only last N per user
            await conn.execute(
                """DELETE FROM history WHERE user_id = ? AND id NOT IN (
                    SELECT id FROM (SELECT id FROM history WHERE user_id = ? ORDER BY id DESC LIMIT ?)
                )""",
                (user_id, user_id, config.HISTORY_MAX_ITEMS),
            )
        await conn.commit()


async def get_history(user_id: int, limit: int | None = None) -> list[dict]:
    """Return list of {category, text} ordered by newest first."""
    limit = limit or config.HISTORY_MAX_ITEMS
    async with aiosqlite.connect(_db_path()) as conn:
        conn.row_factory = aiosqlite.Row
        async with conn.execute(
            """SELECT category, prompt_preview FROM history
               WHERE user_id = ? ORDER BY id DESC LIMIT ?""",
            (user_id, limit),
        ) as cur:
            rows = await cur.fetchall()
    return [{"category": r["category"], "text": r["prompt_preview"]} for r in rows]
