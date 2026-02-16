#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

LOG_FILE="$SCRIPT_DIR/transcribe_bot.log"
PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"

if pgrep -f "[l]ocal_telegram_bot.py" >/dev/null 2>&1; then
  echo "local_telegram_bot.py уже запущен."
  exit 0
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Не найден Python в .venv: $PYTHON_BIN"
  echo "Создайте окружение и установите зависимости:"
  echo "  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi

nohup "$PYTHON_BIN" "$SCRIPT_DIR/local_telegram_bot.py" >> "$LOG_FILE" 2>&1 < /dev/null &
BOT_PID=$!

echo "Бот запущен. PID: $BOT_PID"
echo "Лог: $LOG_FILE"
