# Telegram → LM Studio (openai/gpt-oss-20b) Bot with Global Queue

Это простой Telegram-бот, который ставит **все входящие сообщения от всех пользователей** в **одну очередь** и последовательно отправляет их в LM Studio API (`/v1/chat/completions`).

- Глобальная очередь: один воркер обрабатывает запросы по FIFO, чтобы не перегружать LLM.
- Поддержка истории диалога per‑chat (обрезается до N последних сообщений).
- Команды для управления: `/start`, `/help`, `/reset`, `/model`, `/temp`, `/maxtokens`, `/queue`.
- Готовый Docker/Compose деплой **одной командой**.

> По умолчанию бот стучится в `http://host.docker.internal:1234/v1` — это удобно, когда LM Studio запущен на хосте, а бот — в контейнере.

## Быстрый старт

1) Установите и запустите LM Studio Server (или любой совместимый OpenAI‑совместимый сервер) на хост‑машине на `:1234`  
   Убедитесь, что эндпоинт отвечает на `http://localhost:1234/v1/chat/completions` (как в LM Studio).

2) Скопируйте `.env.example` → `.env` и заполните токен бота:
```bash
cp .env.example .env
# откройте .env и вставьте TELEGRAM_BOT_TOKEN=...
```

3) Запустите бота через Docker Compose:
```bash
docker compose up -d --build
```

Бот поднимется и начнёт принимать сообщения.

## Команды

- `/start` — краткая справка.
- `/help` — список команд.
- `/reset` — очистить историю текущего чата.
- `/model` — показать текущую модель. `/model <name>` — установить модель, например `/model openai/gpt-oss-20b`.
- `/temp` — показать текущую температуру. `/temp 0.7` — задать значение.
- `/maxtokens` — показать текущее ограничение. `/maxtokens 1024` — задать значение (или `-1` для безлимитного, если поддерживается сервером).
- `/queue` — показать размер очереди и ваше место в ней.

## Переменные окружения

Создайте файл `.env` (см. пример в `.env.example`):

```
TELEGRAM_BOT_TOKEN=...            # токен Telegram-бота
LLM_BASE_URL=http://host.docker.internal:1234/v1
LLM_MODEL=openai/gpt-oss-20b
SYSTEM_PROMPT=You are a helpful assistant.
TEMPERATURE=0.7
MAX_TOKENS=-1                     # -1 для "без лимита" (как в LM Studio)
REQUEST_TIMEOUT=180               # таймаут запроса к LLM, сек
MAX_HISTORY_TURNS=12              # сколько последних сообщений истории хранить для каждого чата
```

> Если бот работает на Linux и `host.docker.internal` недоступен, вы можете либо оставить настройку `extra_hosts` в compose (она уже добавлена), либо заменить `LLM_BASE_URL` на `http://172.17.0.1:1234/v1` или включить `network_mode: host` (см. комментарий в `docker-compose.yml`).

## Стек

- Python 3.11, `python-telegram-bot` 21.x (async), `aiohttp`.
- Один глобальный `asyncio.Queue` и воркер, который берёт задачи по одной.

## Локальный запуск без Docker (опционально)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r app/requirements.txt
export $(cat .env | xargs) # или задать переменные окружения вручную
python app/bot.py
```

## Полезное

- Если хотите строгую последовательность **даже без очереди** (например, 1 запрос на процесс), оставьте как есть: воркер — один. Для параллелизма можно сделать несколько воркеров, но **очередь сохранит порядок**, а в worker добавить семафор доступа к LLM.
- История диалога хранится в памяти процесса. Перезапуск бота очистит историю.
