#!/usr/bin/env python3
import asyncio
import json
import logging
import os
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

import aiohttp
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("tg-lmstudio-queue-bot")

# ---------- Config from env ----------
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://host.docker.internal:1234/v1")
LLM_MODEL_DEFAULT = os.environ.get("LLM_MODEL", "openai/gpt-oss-20b")
SYSTEM_PROMPT_DEFAULT = os.environ.get("SYSTEM_PROMPT", "You are a helpful assistant.")
TEMPERATURE_DEFAULT = float(os.environ.get("TEMPERATURE", "0.7"))
MAX_TOKENS_DEFAULT = int(os.environ.get("MAX_TOKENS", "-1"))
REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "180"))
MAX_HISTORY_TURNS = int(os.environ.get("MAX_HISTORY_TURNS", "12"))  # count of messages, not pairs

CHAT_COMPLETIONS_URL = LLM_BASE_URL.rstrip("/") + "/chat/completions"
MODELS_URL = LLM_BASE_URL.rstrip("/") + "/models"

# ---------- Simple in-memory store ----------
# Per-chat history
History = Deque[Dict[str, str]]
histories: Dict[int, History] = defaultdict(lambda: deque(maxlen=MAX_HISTORY_TURNS))

# Per-chat settings
chat_model: Dict[int, str] = defaultdict(lambda: LLM_MODEL_DEFAULT)
chat_temperature: Dict[int, float] = defaultdict(lambda: TEMPERATURE_DEFAULT)
chat_max_tokens: Dict[int, int] = defaultdict(lambda: MAX_TOKENS_DEFAULT)

# App/global cache
models_cache: List[str] = []  # simple cache of model ids

# ---------- Queue types ----------
@dataclass
class Task:
    chat_id: int
    user_id: int
    text: str
    ack_message_id: Optional[int] = None  # message to edit with status
    position_at_enqueue: int = 0

global_queue: "asyncio.Queue[Task]" = asyncio.Queue()

async def llm_request(
    session: aiohttp.ClientSession,
    messages: List[Dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
    async with session.post(CHAT_COMPLETIONS_URL, json=payload, timeout=timeout) as resp:
        if resp.status != 200:
            text = await resp.text()
            raise RuntimeError(f"LLM HTTP {resp.status}: {text}")
        data = await resp.json()
    # OpenAI-style response
    try:
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"Unexpected LLM response format: {e}; data={json.dumps(data)[:500]}")

async def worker(app: Application) -> None:
    """Single worker that processes tasks sequentially."""
    session = aiohttp.ClientSession()
    try:
        while True:
            task: Task = await global_queue.get()
            try:
                # Update status: started
                pending = global_queue.qsize()
                try:
                    if task.ack_message_id:
                        await app.bot.edit_message_text(
                            chat_id=task.chat_id,
                            message_id=task.ack_message_id,
                            text=f"🟡 Обрабатываю (в очереди ещё: {pending})…",
                        )
                except Exception:
                    pass

                # Build messages: system + history + user
                history = list(histories[task.chat_id])
                model = chat_model[task.chat_id]
                temperature = chat_temperature[task.chat_id]
                max_tokens = chat_max_tokens[task.chat_id]

                messages = [{"role": "system", "content": SYSTEM_PROMPT_DEFAULT}]
                messages.extend(history)
                messages.append({"role": "user", "content": task.text})

                # Call LLM
                reply = await llm_request(
                    session=session,
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                # Save history
                histories[task.chat_id].append({"role": "user", "content": task.text})
                histories[task.chat_id].append({"role": "assistant", "content": reply})

                # Send answer (edit ack or send new)
                if task.ack_message_id:
                    try:
                        await app.bot.edit_message_text(
                            chat_id=task.chat_id,
                            message_id=task.ack_message_id,
                            text=reply,
                            parse_mode=ParseMode.MARKDOWN,
                            disable_web_page_preview=True,
                        )
                    except Exception:
                        # Fallback: send a new message if edit failed
                        await app.bot.send_message(
                            chat_id=task.chat_id,
                            text=reply,
                            parse_mode=ParseMode.MARKDOWN,
                            disable_web_page_preview=True,
                        )
                else:
                    await app.bot.send_message(
                        chat_id=task.chat_id,
                        text=reply,
                        parse_mode=ParseMode.MARKDOWN,
                        disable_web_page_preview=True,
                    )

            except Exception as e:
                log.exception("Worker error: %s", e)
                err_text = f"❌ Ошибка обработки: {e}"
                if task.ack_message_id:
                    try:
                        await app.bot.edit_message_text(
                            chat_id=task.chat_id,
                            message_id=task.ack_message_id,
                            text=err_text,
                        )
                    except Exception:
                        await app.bot.send_message(chat_id=task.chat_id, text=err_text)
                else:
                    await app.bot.send_message(chat_id=task.chat_id, text=err_text)
            finally:
                global_queue.task_done()
    finally:
        await session.close()

# ---------- Models utils ----------
async def fetch_models(timeout_sec: int = 8) -> List[str]:
    """Fetch models list from LLM server (OpenAI-compatible /models)."""
    timeout = aiohttp.ClientTimeout(total=timeout_sec)
    async with aiohttp.ClientSession(timeout=timeout) as s:
        async with s.get(MODELS_URL) as r:
            if r.status != 200:
                text = await r.text()
                raise RuntimeError(f"GET /models HTTP {r.status}: {text}")
            data = await r.json()
    # OpenAI format: {"data":[{"id":"...","object":"model",...}, ...]}
    ids: List[str] = []
    try:
        for item in data.get("data", []):
            mid = item.get("id")
            if isinstance(mid, str):
                ids.append(mid)
    except Exception as e:
        raise RuntimeError(f"Unexpected /models format: {e}; data={json.dumps(data)[:500]}")
    return ids

def build_models_keyboard(models: List[str], current: str) -> InlineKeyboardMarkup:
    """Build inline keyboard with model choices (2 per row), mark current."""
    buttons: List[List[InlineKeyboardButton]] = []
    row: List[InlineKeyboardButton] = []
    for m in models:
        label = f"✅ {m}" if m == current else m
        row.append(InlineKeyboardButton(text=label, callback_data=f"set_model|{m}"))
        if len(row) == 2:
            buttons.append(row)
            row = []
    if row:
        buttons.append(row)
    # control row
    buttons.append([InlineKeyboardButton(text="🔄 Обновить список", callback_data="models_refresh")])
    return InlineKeyboardMarkup(buttons)

# ---------- Handlers ----------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (
        "Привет! Я — обёртка над LM Studio API с **глобальной очередью**.\n\n"
        "Просто напишите сообщение — оно попадёт в очередь и будет обработано по порядку.\n\n"
        "Команды:\n"
        "/help — справка\n"
        "/reset — очистить историю\n"
        "/model [name] — показать/установить модель (по умолчанию: {model})\n"
        "/models — получить список моделей и переключиться кнопкой\n"
        "/temp [value] — показать/установить температуру\n"
        "/maxtokens [value] — показать/установить лимит токенов\n"
        "/queue — посмотреть размер очереди и вашу позицию\n"
    ).format(model=LLM_MODEL_DEFAULT)
    await update.message.reply_text(text)

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "/reset — очистить историю\n"
        "/model [name] — показать/установить модель\n"
        "/models — список моделей\n"
        "/temp [value] — показать/установить температуру\n"
        "/maxtokens [value] — показать/установить лимит токенов\n"
        "/queue — очередь\n"
    )

async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    cid = update.effective_chat.id
    histories[cid].clear()
    await update.message.reply_text("🧹 История очищена.")

async def cmd_model(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    global models_cache
    cid = update.effective_chat.id
    if context.args:
        name = " ".join(context.args).strip()
        chat_model[cid] = name
        await update.message.reply_text(f"🔧 Модель обновлена: `{name}`", parse_mode=ParseMode.MARKDOWN)
    else:
        # No args: show current + keyboard with models (from cache or fetch)
        current = chat_model[cid]
        models = models_cache.copy()
        if not models:
            try:
                models = await fetch_models()
                models_cache = models
            except Exception as e:
                await update.message.reply_text(
                    f"Текущая модель: {current}\n⚠️ Не удалось получить список моделей: {e}"
                )
                return
        kb = build_models_keyboard(models, current)
        await update.message.reply_text(f"Текущая модель: {current}\nВыберите модель:", reply_markup=kb)

async def cmd_models(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Fetch and show models with inline keyboard for quick switching."""
    global models_cache
    try:
        models = await fetch_models()
        models_cache = models
    except Exception as e:
        await update.message.reply_text(f"❌ Не удалось получить список моделей: {e}")
        return

    cid = update.effective_chat.id
    current = chat_model[cid]
    if not models:
        await update.message.reply_text("Сервер вернул пустой список моделей.")
        return

    kb = build_models_keyboard(models, current)
    await update.message.reply_text(
        f"Доступные модели: {len(models)}\nТекущая: {current}\nВыберите модель:",
        reply_markup=kb,
    )

async def callbacks(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle inline button presses for model selection and refresh."""
    global models_cache
    query = update.callback_query
    if not query:
        return
    data = (query.data or "").strip()
    cid = update.effective_chat.id

    try:
        if data.startswith("set_model|"):
            new_model = data.split("|", 1)[1]
            chat_model[cid] = new_model
            current = new_model

            # Rebuild keyboard with updated current marking
            models = models_cache or []
            if not models:
                try:
                    models = await fetch_models()
                    models_cache = models
                except Exception:
                    models = []

            kb = build_models_keyboard(models or [new_model], current)
            await query.answer(f"Модель переключена: {new_model}", show_alert=False)
            base_text = f"Текущая модель: {current}\nВыберите модель:"
            try:
                await query.edit_message_text(base_text, reply_markup=kb)
            except Exception:
                try:
                    await query.edit_message_reply_markup(reply_markup=kb)
                except Exception:
                    pass

        elif data == "models_refresh":
            try:
                models = await fetch_models()
                models_cache = models
                current = chat_model[cid]
                kb = build_models_keyboard(models, current)
                await query.answer("Список обновлён.")
                await query.edit_message_reply_markup(reply_markup=kb)
            except Exception:
                await query.answer("Ошибка обновления.", show_alert=True)
        else:
            await query.answer()
    except Exception as e:
        log.exception("Callback error: %s", e)
        try:
            await query.answer("Ошибка обработки кнопки.", show_alert=True)
        except Exception:
            pass

async def cmd_temp(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    cid = update.effective_chat.id
    if context.args:
        try:
            val = float(context.args[0])
        except Exception:
            await update.message.reply_text("Укажите число, например: /temp 0.7")
            return
        chat_temperature[cid] = val
        await update.message.reply_text(f"🔧 Температура обновлена: {val}")
    else:
        await update.message.reply_text(f"Текущая температура: {chat_temperature[cid]}")

async def cmd_maxtokens(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    cid = update.effective_chat.id
    if context.args:
        try:
            val = int(context.args[0])
        except Exception:
            await update.message.reply_text("Укажите целое число, например: /maxtokens 1024 (или -1)")
            return
        chat_max_tokens[cid] = val
        await update.message.reply_text(f"🔧 Лимит токенов обновлён: {val}")
    else:
        await update.message.reply_text(f"Текущий лимит токенов: {chat_max_tokens[cid]}")

async def cmd_queue(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    size = global_queue.qsize()
    await update.message.reply_text(f"В очереди сейчас: {size}")

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return
    if not TELEGRAM_BOT_TOKEN:
        await update.message.reply_text("❌ Не задан TELEGRAM_BOT_TOKEN")
        return

    text = update.message.text.strip()
    cid = update.effective_chat.id
    uid = update.effective_user.id

    position = global_queue.qsize() + 1
    ack = await update.message.reply_text(f"📝 Добавлено в очередь. Ваша позиция: {position}")
    task = Task(
        chat_id=cid,
        user_id=uid,
        text=text,
        ack_message_id=ack.message_id,
        position_at_enqueue=position,
    )
    await global_queue.put(task)

async def on_post_init(app: Application) -> None:
    global models_cache
    # start the single worker
    app.bot_data["worker_task"] = asyncio.create_task(worker(app))
    log.info("Worker started.")
    # Optional: prefetch models (non-fatal)
    try:
        m = await fetch_models(timeout_sec=5)
        models_cache = m
        log.info("Fetched %d models.", len(m))
    except Exception as e:
        log.warning("Models prefetch failed: %s", e)
    # Optional: test connectivity to LM Studio (non-fatal)
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(LLM_BASE_URL.rstrip("/") + "/models", timeout=5) as r:
                log.info("LM Studio models endpoint status: %s", r.status)
    except Exception as e:
        log.warning("LM Studio connectivity check failed: %s", e)

def main() -> None:
    if not TELEGRAM_BOT_TOKEN:
        raise SystemExit("Please set TELEGRAM_BOT_TOKEN env variable.")

    application = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .post_init(on_post_init)
        .build()
    )

    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("help", cmd_help))
    application.add_handler(CommandHandler("reset", cmd_reset))
    application.add_handler(CommandHandler("model", cmd_model))
    application.add_handler(CommandHandler("models", cmd_models))
    application.add_handler(CommandHandler("temp", cmd_temp))
    application.add_handler(CommandHandler("maxtokens", cmd_maxtokens))
    application.add_handler(CommandHandler("queue", cmd_queue))

    application.add_handler(CallbackQueryHandler(callbacks))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    log.info("Starting bot polling…")
    application.run_polling(close_loop=False)

if __name__ == "__main__":
    main()
