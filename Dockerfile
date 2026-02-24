# Telegram prompt bot — deployable image
FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (better layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code (no .env — pass at runtime)
COPY bot.py config.py utils.py db.py ./

# Run as non-root; /app is writable so SQLite can create bot.db here
RUN useradd -m -u 1000 botuser && chown -R botuser:botuser /app
USER botuser

# DB in /app (writable by botuser). For persistence, mount a volume on /app and set DATABASE_PATH
ENV DATABASE_PATH=/app/bot.db

CMD ["python", "-u", "bot.py"]
