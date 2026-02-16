"""
Constants, enums, env validation, and static config for the prompt bot.
"""
import os
from enum import Enum, auto
from pathlib import Path

from dotenv import load_dotenv

_script_dir = Path(__file__).resolve().parent
_env_path = _script_dir / ".env"
load_dotenv(_env_path, override=True)
load_dotenv(Path.cwd() / ".env", override=True)


# --- Env validation (fail fast at import) ---
def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value or not value.strip():
        raise RuntimeError(
            f"{name} is not set. Add it to .env (path: {_env_path}). "
            "Format: KEY=value, no spaces around =."
        )
    return value.strip()


TELEGRAM_BOT_TOKEN = _require_env("TELEGRAM_BOT_TOKEN")
DEEPSEEK_API_KEY = _require_env("DEEPSEEK_API_KEY")
ENV_PATH = _env_path


# --- Enums ---
class ConversationState(int, Enum):
    MAIN_MENU = 0
    AWAITING_DESCRIPTION = auto()
    PROMPT_SHOWN = auto()  # prompt sent; user can Approve or Refine
    AWAITING_REFINEMENT = auto()  # user chose Refine, waiting for extra details


class Category(str, Enum):
    IMAGE = "image"
    CODE = "code"
    VIDEO = "video"
    TEXT = "text"


# --- Telegram / API ---
MAX_MESSAGE_LENGTH = 4096
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_TIMEOUT = 60.0
DEEPSEEK_RETRY_ATTEMPTS = 3
RATE_LIMIT_REQUESTS_PER_MINUTE = 5
HISTORY_MAX_ITEMS = 5


# --- Emojis ---
EMOJI_IMAGE = "🖼"
EMOJI_CODE = "💻"
EMOJI_VIDEO = "🎬"
EMOJI_TEXT = "✍️"

CATEGORY_EMOJI = {
    Category.IMAGE: EMOJI_IMAGE,
    Category.CODE: EMOJI_CODE,
    Category.VIDEO: EMOJI_VIDEO,
    Category.TEXT: EMOJI_TEXT,
}


# --- UI messages ---
MSG_WELCOME = (
    "👋 *Hi!*\n\n"
    "I help you write strong prompts for AI tools.\n"
    "Pick a category, describe what you want — I'll turn it into a ready-to-use prompt.\n\n"
    "👇 _Choose one:_"
)
MSG_HELP = (
    "📖 *Commands*\n\n"
    "• /start — show menu\n"
    "• /help — this message\n"
    "• /cancel — back to menu\n"
    "• /history — last generated prompts\n\n"
    "💡 *How it works*\n"
    "1. Tap a category (Image, Code, Video, Text)\n"
    "2. Describe your idea in a few words or sentences\n"
    "3. Copy the generated prompt and use it in your AI tool"
)
MSG_CHOOSE_CATEGORY = "👇 Choose a category:"
MSG_SENDING = "⏳ Generating your prompt…"
MSG_ERROR_NETWORK = "❌ Network error or timeout. Check your connection and try again."
MSG_ERROR_API = "❌ API error. Check your DeepSeek key and try again."
MSG_ERROR_UNKNOWN = "❌ Something went wrong. Try again or pick another category."
MSG_CANCEL = "↩️ Back to menu. Choose a category:"
MSG_HERE_PROMPT = "✅ _Here's your prompt — copy and use it:_"
MSG_RATE_LIMIT = "⏳ Too many requests. Please wait a minute and try again."
MSG_HISTORY_EMPTY = "📭 No generated prompts yet. Use the menu to create one."
MSG_HISTORY_HEADER = "📜 *Last {} prompts:*\n\n"
MSG_BACK = "↩️ Back to menu"
MSG_APPROVE_OR_REFINE = "What next?"
MSG_SEND_REFINEMENT = "✏️ Send your additional details or changes (e.g. add something, make it shorter, change tone):"

# Raw prompt texts (no escaped newlines for use in code)
_PROMPT_IMAGE = (
    f"{EMOJI_IMAGE} *Image prompt*\n\n"
    "Describe what you want in the image (subject, style, mood, details):"
)
_PROMPT_CODE = (
    f"{EMOJI_CODE} *Code prompt*\n\n"
    "Describe the task (language, what the code should do, any constraints):"
)
_PROMPT_VIDEO = (
    f"{EMOJI_VIDEO} *Video prompt*\n\n"
    "Describe the scene or story (action, camera, style, length):"
)
_PROMPT_TEXT = (
    f"{EMOJI_TEXT} *Text prompt*\n\n"
    "Describe what you need (topic, tone, audience, format):"
)

PROMPT_MESSAGES = {
    Category.IMAGE: _PROMPT_IMAGE,
    Category.CODE: _PROMPT_CODE,
    Category.VIDEO: _PROMPT_VIDEO,
    Category.TEXT: _PROMPT_TEXT,
}


# --- System prompts for DeepSeek (unchanged from user's version) ---
SYSTEM_PROMPTS = {
    Category.IMAGE: (
        "You are an expert prompt engineer for AI image generators (Midjourney, DALL-E, Stable Diffusion). "
        "Based on the user's description, create a highly detailed English prompt following this order: "
        "1. Subject (with appearance, action, expression) "
        "2. Environment and background "
        "3. Lighting and colors "
        "4. Style and mood (e.g., cinematic, cyberpunk, minimalist) "
        "5. Technical specs (8k, photorealistic, unreal engine, etc.) "
        "Use precise, descriptive language. Avoid generic terms. "
        "If aspect ratio is not mentioned, assume square (--ar 1:1). "
        "Output ONLY the prompt, no commentary."
    ),
    Category.VIDEO: (
        "You are an expert prompt engineer for AI video generation (Sora, Runway, Pika). "
        "Create a detailed English prompt based on the user's description. Include: "
        "1. Scene and subject (what happens, who/what is in frame) "
        "2. Camera movement and angles (e.g., slow pan, drone shot, close-up) "
        "3. Motion and pacing (fast/slow, smooth/erratic) "
        "4. Lighting, colors, and atmosphere (cinematic, moody, vibrant) "
        "5. Visual style (photorealistic, 3D animation, cyberpunk, etc.) "
        "6. Duration (if not specified, suggest 5-10 seconds) "
        "7. Optional: sound description or mood (if relevant) "
        "Use vivid, cinematic language. Output ONLY the prompt, no explanations."
    ),
    Category.CODE: (
        "You are an expert prompt engineer for code generation. "
        "Transform the user's task into a precise, structured prompt for an AI coding assistant. "
        "Include: "
        "1. Programming language and version (if relevant) "
        "2. Core functionality and features (what the code should do) "
        "3. Input/output examples or expected behavior "
        "4. Libraries, frameworks, and dependencies "
        "5. Constraints (performance, security, compatibility) "
        "6. Code style (PEP8, comments, type hints, etc.) "
        "7. Edge cases and error handling considerations "
        "Be explicit about what the generated code should accomplish. "
        "Output ONLY the prompt, no explanations."
    ),
    Category.TEXT: (
        "You are an expert prompt engineer for text-based AI (ChatGPT, Claude, etc.). "
        "Craft an effective prompt based on the user's request. The prompt should include: "
        "1. Role for the AI (e.g., 'You are a marketing expert') "
        "2. Context and background information "
        "3. Specific task or question "
        "4. Tone and style (formal, casual, persuasive, humorous) "
        "5. Target audience (experts, beginners, children) "
        "6. Desired format (essay, bullet points, table, dialogue) "
        "7. Length constraints (word count, paragraph count) "
        "8. Examples (if helpful) or what to avoid "
        "Make the prompt detailed but concise. Output ONLY the final prompt, no explanations."
    ),
}

# Used when user chooses "Refine" — we send current prompt + user's refinement
REFINEMENT_SYSTEM_PROMPT = (
    "You are refining an existing prompt. You will receive: "
    "1) The current prompt, 2) The user's requested changes or additions. "
    "Output ONLY the improved full prompt that incorporates the user's feedback. "
    "No commentary or explanation."
)
