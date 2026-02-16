#!/usr/bin/env python3
"""
Telegram bot for generating prompts via DeepSeek API.
Uses config (constants, enums), async DeepSeek with retry, rate limit, history.
"""

import asyncio
import logging
from functools import lru_cache

from openai import AsyncOpenAI
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import BadRequest
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)

import config
from config import (
    Category,
    ConversationState,
    PROMPT_MESSAGES,
    REFINEMENT_SYSTEM_PROMPT,
    SYSTEM_PROMPTS,
)
from utils import RateLimiter, split_into_chunks

# --- Logging (no sensitive data in logs) ---
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def _category_keyboard(include_help: bool = True) -> InlineKeyboardMarkup:
    rows = [
        [
            InlineKeyboardButton(
                f"{config.EMOJI_IMAGE} Image",
                callback_data=Category.IMAGE.value,
            ),
            InlineKeyboardButton(
                f"{config.EMOJI_CODE} Code",
                callback_data=Category.CODE.value,
            ),
        ],
        [
            InlineKeyboardButton(
                f"{config.EMOJI_VIDEO} Video",
                callback_data=Category.VIDEO.value,
            ),
            InlineKeyboardButton(
                f"{config.EMOJI_TEXT} Text",
                callback_data=Category.TEXT.value,
            ),
        ],
    ]
    if include_help:
        rows.append([InlineKeyboardButton("❓ Help", callback_data="help")])
    return InlineKeyboardMarkup(rows)


@lru_cache(maxsize=1)
def get_category_keyboard() -> InlineKeyboardMarkup:
    """Cached main menu keyboard (categories + Help)."""
    return _category_keyboard(include_help=True)


def get_awaiting_keyboard() -> InlineKeyboardMarkup:
    """Keyboard when waiting for description: Back button only."""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(config.MSG_BACK, callback_data="back")],
    ])


def get_approve_refine_keyboard() -> InlineKeyboardMarkup:
    """Keyboard after prompt is shown: Approve or Refine."""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ Approve", callback_data="approve"),
            InlineKeyboardButton("✏️ Refine", callback_data="refine"),
        ],
    ])


async def send_main_menu(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    *,
    text: str | None = None,
) -> int:
    msg = text or config.MSG_WELCOME
    if update.message:
        await update.message.reply_text(
            msg,
            reply_markup=get_category_keyboard(),
            parse_mode="Markdown",
        )
    else:
        chat_id = update.effective_chat.id if update.effective_chat else None
        if chat_id:
            await context.bot.send_message(
                chat_id=chat_id,
                text=msg,
                reply_markup=get_category_keyboard(),
                parse_mode="Markdown",
            )
    return ConversationState.MAIN_MENU


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    return await send_main_menu(update, context)


async def entry_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Entry point when user taps an inline button but conversation was lost (e.g. after bot restart).
    Shows main menu so the bot works without clearing chat.
    """
    query = update.callback_query
    if query:
        await query.answer()
    chat_id = (query.message.chat_id if query and query.message else None) or (
        update.effective_chat.id if update.effective_chat else None
    )
    if chat_id:
        await context.bot.send_message(
            chat_id=chat_id,
            text=config.MSG_WELCOME,
            reply_markup=get_category_keyboard(),
            parse_mode="Markdown",
        )
    return ConversationState.MAIN_MENU


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    target = update.message or (update.callback_query and update.callback_query.message)
    chat_id = update.effective_chat.id if update.effective_chat else None
    if not chat_id:
        return
    if target:
        await target.reply_text(config.MSG_HELP, parse_mode="Markdown")
    else:
        await context.bot.send_message(
            chat_id=chat_id,
            text=config.MSG_HELP,
            parse_mode="Markdown",
        )


async def call_deepseek(system_prompt: str, user_message: str) -> str:
    """Async DeepSeek call with retries. Non-blocking for the event loop."""
    client = AsyncOpenAI(
        api_key=config.DEEPSEEK_API_KEY,
        base_url=config.DEEPSEEK_BASE_URL,
    )
    last_error = None
    for attempt in range(config.DEEPSEEK_RETRY_ATTEMPTS):
        try:
            response = await client.chat.completions.create(
                model=config.DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                timeout=config.DEEPSEEK_TIMEOUT,
            )
            content = response.choices[0].message.content
            if not content or not content.strip():
                raise ValueError("Empty response from DeepSeek")
            return content.strip()
        except Exception as e:
            last_error = e
            if attempt < config.DEEPSEEK_RETRY_ATTEMPTS - 1:
                await asyncio.sleep(2 ** attempt)
    raise last_error


rate_limiter = RateLimiter(config.RATE_LIMIT_REQUESTS_PER_MINUTE)


async def category_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    if query is None:
        return ConversationState.MAIN_MENU
    await query.answer()
    data = query.data or ""
    try:
        category = Category(data)
    except ValueError:
        return ConversationState.MAIN_MENU
    context.user_data["category"] = category.value
    logger.info("User %s chose category: %s", query.from_user.id if query.from_user else 0, data)
    prompt_text = PROMPT_MESSAGES[category]
    chat_id = (query.message.chat_id if query.message else None) or (
        update.effective_chat.id if update.effective_chat else None
    )
    if chat_id is None:
        return ConversationState.MAIN_MENU
    await context.bot.send_message(
        chat_id=chat_id,
        text=prompt_text,
        parse_mode="Markdown",
        reply_markup=get_awaiting_keyboard(),
    )
    return ConversationState.AWAITING_DESCRIPTION


async def help_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    if query:
        await query.answer()
    await cmd_help(update, context)
    return ConversationState.MAIN_MENU


async def back_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    if query:
        await query.answer()
    chat_id = (query.message.chat_id if query and query.message else None) or (
        update.effective_chat.id if update.effective_chat else None
    )
    if chat_id:
        await context.bot.send_message(
            chat_id=chat_id,
            text=config.MSG_CHOOSE_CATEGORY,
            reply_markup=get_category_keyboard(),
            parse_mode="Markdown",
        )
    return ConversationState.MAIN_MENU


async def approve_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """User accepted the prompt — back to main menu."""
    query = update.callback_query
    if query:
        await query.answer()
    chat_id = (query.message.chat_id if query and query.message else None) or (
        update.effective_chat.id if update.effective_chat else None
    )
    if chat_id:
        await context.bot.send_message(
            chat_id=chat_id,
            text=config.MSG_CHOOSE_CATEGORY,
            reply_markup=get_category_keyboard(),
            parse_mode="Markdown",
        )
    return ConversationState.MAIN_MENU


async def refine_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """User wants to refine — ask for additional details."""
    query = update.callback_query
    if query:
        await query.answer()
    chat_id = (query.message.chat_id if query and query.message else None) or (
        update.effective_chat.id if update.effective_chat else None
    )
    if chat_id:
        await context.bot.send_message(
            chat_id=chat_id,
            text=config.MSG_SEND_REFINEMENT,
            reply_markup=get_awaiting_keyboard(),
            parse_mode="Markdown",
        )
    return ConversationState.AWAITING_REFINEMENT


async def handle_description(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id if update.effective_user else 0
    category_value = context.user_data.get("category")
    if not category_value:
        await update.message.reply_text(
            config.MSG_CHOOSE_CATEGORY,
            reply_markup=get_category_keyboard(),
            parse_mode="Markdown",
        )
        return ConversationState.MAIN_MENU
    try:
        category = Category(category_value)
    except ValueError:
        await update.message.reply_text(
            config.MSG_CHOOSE_CATEGORY,
            reply_markup=get_category_keyboard(),
            parse_mode="Markdown",
        )
        return ConversationState.MAIN_MENU

    user_text = (update.message.text or "").strip()
    if not user_text:
        await update.message.reply_text(
            PROMPT_MESSAGES[category],
            parse_mode="Markdown",
            reply_markup=get_awaiting_keyboard(),
        )
        return ConversationState.AWAITING_DESCRIPTION

    if not rate_limiter.is_allowed(user_id):
        await update.message.reply_text(config.MSG_RATE_LIMIT, parse_mode="Markdown")
        return ConversationState.AWAITING_DESCRIPTION

    system_prompt = SYSTEM_PROMPTS[category]
    status_msg = await update.message.reply_text(config.MSG_SENDING, parse_mode="Markdown")

    try:
        result = await call_deepseek(system_prompt, user_text)
        try:
            await status_msg.delete()
        except BadRequest:
            pass

        # History: append (keep last N)
        history = context.user_data.get("history") or []
        history.append({"category": category_value, "text": result[:200] + ("…" if len(result) > 200 else "")})
        context.user_data["history"] = history[-config.HISTORY_MAX_ITEMS :]

        # Send result: header (Markdown), then prompt (plain or word-safe chunks)
        await update.message.reply_text(config.MSG_HERE_PROMPT, parse_mode="Markdown")
        if len(result) <= config.MAX_MESSAGE_LENGTH:
            await update.message.reply_text(result)
        else:
            for chunk in split_into_chunks(result):
                await update.message.reply_text(chunk)

        context.user_data["last_prompt"] = result
        context.user_data["last_category"] = category_value
        context.user_data["original_description"] = user_text
        await update.message.reply_text(
            config.MSG_APPROVE_OR_REFINE,
            reply_markup=get_approve_refine_keyboard(),
            parse_mode="Markdown",
        )
        logger.info("Generated prompt for user %s, category %s", user_id, category_value)
    except Exception as e:
        logger.exception("DeepSeek API error: %s", e)
        err_msg = config.MSG_ERROR_UNKNOWN
        if "timeout" in str(e).lower() or "connection" in str(e).lower():
            err_msg = config.MSG_ERROR_NETWORK
        elif "api_key" in str(e).lower() or "401" in str(e) or "429" in str(e):
            err_msg = config.MSG_ERROR_API
        try:
            await status_msg.edit_text(err_msg, parse_mode="Markdown")
        except Exception:
            await update.message.reply_text(err_msg, parse_mode="Markdown")
        await update.message.reply_text(
            config.MSG_CHOOSE_CATEGORY,
            reply_markup=get_category_keyboard(),
            parse_mode="Markdown",
        )
        return ConversationState.MAIN_MENU

    return ConversationState.PROMPT_SHOWN


async def handle_refinement(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """User sent refinement text — improve the prompt and show again with Approve/Refine."""
    user_id = update.effective_user.id if update.effective_user else 0
    last_prompt = context.user_data.get("last_prompt")
    if not last_prompt:
        await update.message.reply_text(
            config.MSG_CHOOSE_CATEGORY,
            reply_markup=get_category_keyboard(),
            parse_mode="Markdown",
        )
        return ConversationState.MAIN_MENU

    user_text = (update.message.text or "").strip()
    if not user_text:
        await update.message.reply_text(
            config.MSG_SEND_REFINEMENT,
            reply_markup=get_awaiting_keyboard(),
            parse_mode="Markdown",
        )
        return ConversationState.AWAITING_REFINEMENT

    if not rate_limiter.is_allowed(user_id):
        await update.message.reply_text(config.MSG_RATE_LIMIT, parse_mode="Markdown")
        return ConversationState.AWAITING_REFINEMENT

    user_message = f"Current prompt:\n{last_prompt}\n\nUser's requested changes or additions:\n{user_text}"
    status_msg = await update.message.reply_text(config.MSG_SENDING, parse_mode="Markdown")

    try:
        result = await call_deepseek(REFINEMENT_SYSTEM_PROMPT, user_message)
        try:
            await status_msg.delete()
        except BadRequest:
            pass
        context.user_data["last_prompt"] = result
        await update.message.reply_text(config.MSG_HERE_PROMPT, parse_mode="Markdown")
        if len(result) <= config.MAX_MESSAGE_LENGTH:
            await update.message.reply_text(result)
        else:
            for chunk in split_into_chunks(result):
                await update.message.reply_text(chunk)
        await update.message.reply_text(
            config.MSG_APPROVE_OR_REFINE,
            reply_markup=get_approve_refine_keyboard(),
            parse_mode="Markdown",
        )
        logger.info("Refined prompt for user %s", user_id)
    except Exception as e:
        logger.exception("Refinement API error: %s", e)
        err_msg = config.MSG_ERROR_UNKNOWN
        if "timeout" in str(e).lower() or "connection" in str(e).lower():
            err_msg = config.MSG_ERROR_NETWORK
        elif "api_key" in str(e).lower() or "401" in str(e) or "429" in str(e):
            err_msg = config.MSG_ERROR_API
        try:
            await status_msg.edit_text(err_msg, parse_mode="Markdown")
        except Exception:
            await update.message.reply_text(err_msg, parse_mode="Markdown")
        await update.message.reply_text(
            config.MSG_APPROVE_OR_REFINE,
            reply_markup=get_approve_refine_keyboard(),
            parse_mode="Markdown",
        )
    return ConversationState.PROMPT_SHOWN


async def cmd_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text(
        config.MSG_CANCEL,
        reply_markup=get_category_keyboard(),
        parse_mode="Markdown",
    )
    return ConversationState.MAIN_MENU


async def cmd_history(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    history = context.user_data.get("history") or []
    if not history:
        await update.message.reply_text(
            config.MSG_HISTORY_EMPTY,
            parse_mode="Markdown",
        )
        return
    lines = []
    for i, item in enumerate(reversed(history), 1):
        cat = item.get("category", "?")
        preview = (item.get("text") or "")[:150]
        lines.append(f"{i}. *{cat}* — {preview}…" if len(preview) >= 150 else f"{i}. *{cat}* — {preview}")
    text = config.MSG_HISTORY_HEADER.format(len(history)) + "\n".join(lines)
    await update.message.reply_text(text, parse_mode="Markdown")


def main() -> None:
    application = (
        Application.builder()
        .token(config.TELEGRAM_BOT_TOKEN)
        .build()
    )

    application.add_handler(CommandHandler("help", cmd_help))
    application.add_handler(CommandHandler("history", cmd_history))

    async def main_menu_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        await update.message.reply_text(
            config.MSG_CHOOSE_CATEGORY,
            reply_markup=get_category_keyboard(),
            parse_mode="Markdown",
        )
        return ConversationState.MAIN_MENU

    category_pattern = "|".join(c.value for c in Category)
    main_menu_handlers = [
        CallbackQueryHandler(category_callback, pattern=f"^({category_pattern})$"),
        CallbackQueryHandler(help_callback, pattern="^help$"),
        MessageHandler(filters.TEXT & ~filters.COMMAND, main_menu_text),
    ]
    awaiting_handlers = [
        CallbackQueryHandler(back_callback, pattern="^back$"),
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_description),
    ]
    prompt_shown_handlers = [
        CallbackQueryHandler(approve_callback, pattern="^approve$"),
        CallbackQueryHandler(refine_callback, pattern="^refine$"),
    ]
    awaiting_refinement_handlers = [
        CallbackQueryHandler(back_callback, pattern="^back$"),
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_refinement),
    ]

    conv_handler = ConversationHandler(
        entry_points=[
            CommandHandler("start", cmd_start),
            MessageHandler(filters.TEXT & ~filters.COMMAND, cmd_start),
            CallbackQueryHandler(entry_callback),
        ],
        states={
            ConversationState.MAIN_MENU: main_menu_handlers,
            ConversationState.AWAITING_DESCRIPTION: awaiting_handlers,
            ConversationState.PROMPT_SHOWN: prompt_shown_handlers,
            ConversationState.AWAITING_REFINEMENT: awaiting_refinement_handlers,
        },
        fallbacks=[
            CallbackQueryHandler(category_callback, pattern=f"^({category_pattern})$"),
            CallbackQueryHandler(help_callback, pattern="^help$"),
            CallbackQueryHandler(back_callback, pattern="^back$"),
            CallbackQueryHandler(approve_callback, pattern="^approve$"),
            CallbackQueryHandler(refine_callback, pattern="^refine$"),
            CommandHandler("cancel", cmd_cancel),
            CommandHandler("start", cmd_start),
        ],
    )
    application.add_handler(conv_handler)

    async def on_error(_update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        logger.exception("Unhandled error: %s", context.error)

    application.add_error_handler(on_error)
    logger.info("Bot starting (long-polling)")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
