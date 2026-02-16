# Telegram prompt bot — deployable image
FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (better layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code (no .env — pass at runtime)
COPY bot.py config.py utils.py ./

# Run as non-root
RUN useradd -m -u 1000 botuser && chown -R botuser:botuser /app
USER botuser

CMD ["python", "-u", "bot.py"]
