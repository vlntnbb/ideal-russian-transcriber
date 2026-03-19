from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import tempfile
import time
import json
import threading
import io
import datetime
import sys
import warnings
import uuid
import html as _html
import re
import secrets
import smtplib
import ssl
import subprocess
import threading
import multiprocessing
import wave
from email.message import EmailMessage
from typing import Optional

from dotenv import load_dotenv
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction, ChatType, ParseMode
from telegram.error import BadRequest, RetryAfter
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes, MessageHandler, filters
import httpx

from snapscript.core.audio_processor import (
    ASRSegment,
    AudioProcessor,
    GigaAMTranscriptionService,
    TranscriptionCancelled,
    TranscriptionService,
    gigaam_transcribe_worker,
)
from snapscript.utils.logging_utils import setup_logging
from transcribe_core.markdown import render_markdown

# Prevent duplicate local_telegram_bot polling instances (avoids Telegram getUpdates Conflict).
def _acquire_bot_lock(lock_path: str) -> None:
    import fcntl

    os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)
    lock_file = open(lock_path, "a+", encoding="utf-8")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        raise SystemExit(
            f"Another bot instance seems to be running (lock: {lock_path}). Stop it before starting a new one."
        ) from exc
    # Keep FD alive for the whole process.
    globals()["_BOT_LOCK_FD"] = lock_file


TELEGRAM_TEXT_LIMIT = 4096
STATS_PATH = os.path.join(os.path.dirname(__file__), "asr_stats.json")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_USAGE_LOG_PATH = os.path.join(os.path.dirname(__file__), "usage_sessions.jsonl")
DEFAULT_ACTIVE_SESSIONS_PATH = os.path.join(os.path.dirname(__file__), "active_sessions.json")
AUTH_STATE_PATH = os.path.join(os.path.dirname(__file__), "auth_state.json")
DEFAULT_GEMINI_SYSTEM_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "gemini_system_prompt.md")

# Email auth: only this domain can authorize users for Gemini access.
DEFAULT_AUTH_DOMAIN = "bbooster.io"
DEFAULT_AUTH_CODE_TTL_SEC = 15 * 60  # 15 minutes
DEFAULT_AUTH_CODE_MIN_WORDS = 6
DEFAULT_AUTH_CODE_MAX_WORDS = 10

# Authorized user LLM preference.
LLM_PROVIDER_GEMINI = "gemini"
LLM_PROVIDER_LOCAL = "local"

# Open-source fallback model (Ollama).
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "deepseek-r1:8b"
DEFAULT_AUDIO_NORMALIZE_KEY = "norm,норм"
DEFAULT_AUDIO_NORMALIZE_FILTER = "adeclick,dynaudnorm=f=500:g=11:p=0.95:m=20"
MEDIA_GROUP_WAIT_SEC = 1.2
MEDIA_GROUP_CHOICE_TTL_SEC = 5 * 60

_USAGE_LOGGER: Optional[logging.Logger] = None

DEFAULT_GEMINI_SYSTEM_PROMPT = """\
вот две транскрибации одного аудио двумя разными моделями. они косячат в разных местах. твоя задача сделать итоговый текст, взяв из каждой модели пропущенные предложения и исправив явные ошибки и убрав повторы. когда какие-то слова будут отличаться выбирай тот вариант, который из контекста выглядит более правильным. если в обоих случаях текст выглядит как с ошибками - исправь исходя из контекста. ни в коем случае не выкидывай никакие смысловые фразы и предложения, тебе нужно соблюсти точность передачи текста. подсвети жирным те части текста, где тебе пришлось исправлять текст (где он отличается между моделями).

обращай внимание на указания для контент-менеджера, что надо найти, исправить и уточнить по этому тексту если они есть. в конце отчитайся какие действия с текстами ты сделал

вот такой формат выдачи (шаблон)

### Итоговый текст:

### Примечания для контент-менеджера:

### Отчет о проделанных действиях:
"""


def _read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _get_system_prompt() -> tuple[str, str]:
    """
    Returns (prompt, source) where source is one of: file/env/default.
    Priority: GEMINI_SYSTEM_PROMPT_FILE -> GEMINI_SYSTEM_PROMPT -> default constant.
    """
    p = (os.environ.get("GEMINI_SYSTEM_PROMPT_FILE") or "").strip()
    candidate = p or DEFAULT_GEMINI_SYSTEM_PROMPT_PATH
    try:
        if candidate and os.path.exists(candidate):
            out = (_read_text_file(candidate) or "").strip()
            if out:
                return out, f"file:{candidate}"
    except Exception:
        logging.getLogger("local_telegram_bot").exception("Failed to read system prompt file: %s", candidate)

    env_prompt = (os.environ.get("GEMINI_SYSTEM_PROMPT") or "").strip()
    if env_prompt:
        return env_prompt, "env"
    return DEFAULT_GEMINI_SYSTEM_PROMPT.strip(), "default"


def _audio_normalize_key_tokens() -> list[str]:
    raw = (os.environ.get("AUDIO_NORMALIZE_KEY") or "").strip()
    if not raw:
        raw = DEFAULT_AUDIO_NORMALIZE_KEY
    # Allow comma-separated keys.
    keys = [t.strip().lower() for t in raw.split(",") if t.strip()]
    # Convenience aliases: accept both "norm" and "норм".
    if "norm" in keys and "норм" not in keys:
        keys.append("норм")
    if "норм" in keys and "norm" not in keys:
        keys.append("norm")
    return keys


def _audio_normalize_filter() -> str:
    return (os.environ.get("AUDIO_NORMALIZE_FILTER") or DEFAULT_AUDIO_NORMALIZE_FILTER).strip()


def _should_normalize_audio(text: str, *, bot_username: Optional[str]) -> bool:
    if not text or not bot_username:
        return False
    keys = _audio_normalize_key_tokens()
    if not keys:
        return False

    # Find the bot mention and inspect tokens after it.
    pattern = re.compile(rf"@{re.escape(bot_username)}\b(.*)$", re.IGNORECASE | re.DOTALL)
    m = pattern.search(text)
    if not m:
        return False
    tail = (m.group(1) or "").strip()
    if not tail:
        return False
    for tok in re.split(r"\s+", tail):
        tok = (tok or "").strip()
        if not tok:
            continue
        tok = tok.strip(".,;:!?()[]{}<>\"'")
        tok = tok.split("=", 1)[0].split(":", 1)[0].strip().lower()
        if tok in keys:
            return True
    return False


def _fmt_dur(sec: float) -> str:
    sec = max(0, int(sec))
    if sec < 60:
        return f"{sec}s"
    m, s = divmod(sec, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def _truthy_env(name: str, default: bool = True) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw not in {"0", "false", "no", "off", "disabled"}


def _int_env(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    try:
        return int(raw) if raw else int(default)
    except Exception:
        return int(default)


def _bot_concurrent_updates() -> int:
    return max(1, _int_env("BOT_CONCURRENT_UPDATES", 8))


def _asr_concurrency() -> int:
    return max(1, _int_env("ASR_CONCURRENCY", 3))


def _llm_concurrency() -> int:
    return max(1, _int_env("LLM_CONCURRENCY", 2))


_SPELL_WORDS = [
    # Harry Potter-ish (latin-ish) "spell" words, good for manual typing.
    "accio",
    "alohomora",
    "avada",
    "kedavra",
    "crucio",
    "imperio",
    "expelliarmus",
    "stupefy",
    "protego",
    "lumos",
    "nox",
    "expecto",
    "patronum",
    "reducto",
    "sectumsempra",
    "obliviate",
    "wingardium",
    "leviosa",
    "riddikulus",
    "muffliato",
    "finite",
    "incantatem",
    "petrificus",
    "totalus",
    "confundo",
    "diffindo",
    "aguamenti",
    "avis",
    "duro",
    "glacius",
    "silencio",
    "sonorus",
    "quietus",
    "locomotor",
    "morsmordre",
    "reparo",
    "scourgify",
    "tarantallegra",
    "episkey",
    "priori",
    "incarcerous",
    "liberacorpus",
    "legilimens",
    "occlumens",
    "verdimillious",
    "periculum",
    "portus",
    "dissendium",
    "serpensortia",
]

_AUTH_LOCK = threading.Lock()
_AUTH_STATE: Optional[dict] = None


def _auth_enabled() -> bool:
    return _truthy_env("AUTH_ENABLED", False)


def _auth_domain() -> str:
    return (os.environ.get("AUTH_DOMAIN") or DEFAULT_AUTH_DOMAIN).strip().lower()


def _auth_code_ttl_sec() -> int:
    raw = (os.environ.get("AUTH_CODE_TTL_SEC") or "").strip()
    if raw.isdigit():
        return max(60, int(raw))
    raw_min = (os.environ.get("AUTH_CODE_TTL_MIN") or "").strip()
    if raw_min.isdigit():
        return max(1, int(raw_min)) * 60
    return DEFAULT_AUTH_CODE_TTL_SEC


def _auth_code_word_count() -> tuple[int, int]:
    def _int(name: str, default: int) -> int:
        raw = (os.environ.get(name) or "").strip()
        return int(raw) if raw.isdigit() else default

    mn = _int("AUTH_CODE_MIN_WORDS", DEFAULT_AUTH_CODE_MIN_WORDS)
    mx = _int("AUTH_CODE_MAX_WORDS", DEFAULT_AUTH_CODE_MAX_WORDS)
    mn = max(3, mn)
    mx = max(mn, mx)
    return mn, mx


def _auth_state_default() -> dict:
    return {
        "authorized_users": {},  # user_id(str) -> {"email": str, "authorized_at": iso}
        "whitelisted_chats": {},  # chat_id(str) -> {"added_by": user_id, "added_at": iso}
        "pending": {},  # user_id(str) -> {"email": str, "code": str, "expires_at": iso, "attempts": int}
        "prompted_users": {},  # user_id(str) -> iso
    }


def _load_auth_state() -> dict:
    global _AUTH_STATE
    if _AUTH_STATE is not None:
        return _AUTH_STATE
    with _AUTH_LOCK:
        if _AUTH_STATE is not None:
            return _AUTH_STATE
        state = _auth_state_default()
        try:
            if os.path.exists(AUTH_STATE_PATH):
                with open(AUTH_STATE_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f) or {}
                if isinstance(data, dict):
                    state.update({k: v for k, v in data.items() if k in state and isinstance(v, dict)})
        except Exception:
            logging.getLogger("local_telegram_bot").exception("Failed to load auth state")
        _AUTH_STATE = state
        return state


def _save_auth_state(state: dict) -> None:
    try:
        tmp = AUTH_STATE_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(tmp, AUTH_STATE_PATH)
    except Exception:
        logging.getLogger("local_telegram_bot").exception("Failed to save auth state")


def _now_utc_iso() -> str:
    return datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).isoformat()


def _normalize_code(text: str) -> str:
    s = (text or "").strip().lower()
    # Keep letters/numbers and collapse everything else to spaces.
    s = re.sub(r"[^a-z0-9а-яё]+", " ", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"\\s+", " ", s).strip()
    return s


def _make_spell_code() -> str:
    rng = secrets.SystemRandom()
    mn, mx = _auth_code_word_count()
    n = rng.randint(mn, mx)
    if len(_SPELL_WORDS) >= n:
        words = rng.sample(_SPELL_WORDS, k=n)
    else:
        words = [rng.choice(_SPELL_WORDS) for _ in range(n)]
    rng.shuffle(words)
    return " ".join(words)


def _auth_is_user_authorized(user_id: int) -> bool:
    st = _load_auth_state()
    return str(int(user_id)) in (st.get("authorized_users") or {})


def _auth_is_chat_whitelisted(chat_id: int) -> bool:
    st = _load_auth_state()
    return str(int(chat_id)) in (st.get("whitelisted_chats") or {})


def _auth_can_use_gemini(user_id: int, chat_id: int) -> bool:
    if not _auth_enabled():
        return True
    return _auth_is_user_authorized(user_id) or _auth_is_chat_whitelisted(chat_id)


def _normalize_llm_provider(raw: str) -> Optional[str]:
    token = (raw or "").strip().lower()
    if token in {"gemini", "google", "g"}:
        return LLM_PROVIDER_GEMINI
    if token in {"local", "ollama", "l"}:
        return LLM_PROVIDER_LOCAL
    return None


def _auth_get_user_llm_provider(user_id: int) -> str:
    st = _load_auth_state()
    rec = (st.get("authorized_users") or {}).get(str(int(user_id)))
    if not isinstance(rec, dict):
        return LLM_PROVIDER_GEMINI
    provider = _normalize_llm_provider(str(rec.get("llm_provider") or ""))
    return provider or LLM_PROVIDER_GEMINI


def _auth_set_user_llm_provider(user_id: int, provider: str) -> bool:
    normalized = _normalize_llm_provider(provider)
    if not normalized:
        return False
    uid = str(int(user_id))
    with _AUTH_LOCK:
        st = _load_auth_state()
        rec = (st.get("authorized_users") or {}).get(uid)
        if not isinstance(rec, dict):
            return False
        rec["llm_provider"] = normalized
        rec["llm_provider_updated_at"] = _now_utc_iso()
        _save_auth_state(st)
    return True


def _auth_maybe_whitelist_chat(*, user_id: int, chat_id: int) -> bool:
    """
    If user is authorized and chat isn't whitelisted yet, add it.
    Returns True when chat was newly added.
    """
    if not _auth_enabled():
        return False
    if not _auth_is_user_authorized(user_id):
        return False
    if _auth_is_chat_whitelisted(chat_id):
        return False
    with _AUTH_LOCK:
        st = _load_auth_state()
        if str(int(chat_id)) in st["whitelisted_chats"]:
            return False
        st["whitelisted_chats"][str(int(chat_id))] = {"added_by": int(user_id), "added_at": _now_utc_iso()}
        _save_auth_state(st)
        return True


def _auth_should_prompt_user(user_id: int) -> bool:
    if not _auth_enabled():
        return False
    st = _load_auth_state()
    return str(int(user_id)) not in (st.get("prompted_users") or {})


def _auth_mark_prompted(user_id: int) -> None:
    with _AUTH_LOCK:
        st = _load_auth_state()
        st["prompted_users"][str(int(user_id))] = _now_utc_iso()
        _save_auth_state(st)


def _auth_start_pending(user_id: int, email: str) -> str:
    code = _make_spell_code()
    ttl = _auth_code_ttl_sec()
    expires = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc) + datetime.timedelta(seconds=ttl)
    with _AUTH_LOCK:
        st = _load_auth_state()
        st["pending"][str(int(user_id))] = {
            "email": email,
            "code": code,
            "expires_at": expires.isoformat(),
            "attempts": 0,
        }
        _save_auth_state(st)
    return code


def _auth_has_pending(user_id: int) -> bool:
    st = _load_auth_state()
    return str(int(user_id)) in (st.get("pending") or {})


def _auth_verify_code(user_id: int, text: str) -> tuple[bool, str]:
    """
    Returns (ok, message). On success also persists the authorization.
    """
    domain = _auth_domain()
    uid = str(int(user_id))
    norm = _normalize_code(text)
    with _AUTH_LOCK:
        st = _load_auth_state()
        pending = (st.get("pending") or {}).get(uid)
        if not pending:
            return False, f"Нет активного кода. Отправьте мне вашу почту вида `email@{domain}`."

        expires_at = (pending.get("expires_at") or "").strip()
        try:
            exp = datetime.datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
        except Exception:
            exp = None
        if not exp or exp <= datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc):
            st["pending"].pop(uid, None)
            _save_auth_state(st)
            return False, f"Код истёк. Отправьте мне вашу почту вида `email@{domain}` ещё раз."

        expected = _normalize_code(pending.get("code") or "")
        if norm != expected:
            pending["attempts"] = int(pending.get("attempts") or 0) + 1
            if pending["attempts"] >= 5:
                st["pending"].pop(uid, None)
                _save_auth_state(st)
                return False, f"Слишком много попыток. Отправьте мне вашу почту вида `email@{domain}` ещё раз."
            _save_auth_state(st)
            return False, "Код не совпал. Проверьте слова и порядок (как в письме) и попробуйте ещё раз."

        email = (pending.get("email") or "").strip().lower()
        st["pending"].pop(uid, None)
        st["authorized_users"][uid] = {
            "email": email,
            "authorized_at": _now_utc_iso(),
            "llm_provider": LLM_PROVIDER_GEMINI,
        }
        _save_auth_state(st)
        return True, "✅ Авторизация успешна. Теперь доступна обработка через Gemini.\nПереключение LLM: /llm gemini или /llm local."


def _smtp_send_email(*, to_email: str, subject: str, body: str) -> None:
    host = (os.environ.get("SMTP_HOST") or "").strip()
    if not host:
        raise RuntimeError("SMTP_HOST не задан")
    port = int((os.environ.get("SMTP_PORT") or "").strip() or "587")
    username = (os.environ.get("SMTP_USERNAME") or "").strip()
    password = (os.environ.get("SMTP_PASSWORD") or "").strip()
    from_email = (os.environ.get("SMTP_FROM") or "").strip() or username
    if not from_email:
        raise RuntimeError("SMTP_FROM не задан (и SMTP_USERNAME пуст)")

    use_ssl = _truthy_env("SMTP_SSL", False)
    starttls = _truthy_env("SMTP_STARTTLS", True)

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = to_email
    msg.set_content(body)

    if use_ssl:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(host, port, context=context, timeout=20) as s:
            if username and password:
                s.login(username, password)
            s.send_message(msg)
        return

    with smtplib.SMTP(host, port, timeout=20) as s:
        s.ehlo()
        if starttls:
            context = ssl.create_default_context()
            s.starttls(context=context)
            s.ehlo()
        if username and password:
            s.login(username, password)
        s.send_message(msg)


def _setup_usage_logger() -> Optional[logging.Logger]:
    """
    Separate usage log for sessions (JSON Lines).
    Does not include transcripts/prompts; only metadata and timings.
    """
    global _USAGE_LOGGER
    if _USAGE_LOGGER is not None:
        return _USAGE_LOGGER

    if not _truthy_env("USAGE_LOG_ENABLED", True):
        _USAGE_LOGGER = None
        return None

    path = (os.environ.get("USAGE_LOG_PATH") or "").strip() or DEFAULT_USAGE_LOG_PATH
    try:
        logger = logging.getLogger("usage_sessions")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        _USAGE_LOGGER = logger
        logging.getLogger("local_telegram_bot").info("Usage log enabled: %s", path)
        return logger
    except Exception:
        logging.getLogger("local_telegram_bot").exception("Failed to set up usage logger")
        _USAGE_LOGGER = None
        return None


def _log_usage_session(payload: dict) -> None:
    logger = _USAGE_LOGGER or logging.getLogger("usage_sessions")
    if not logger.handlers:
        return
    try:
        logger.info(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    except Exception:
        # Usage logging must never break bot flow.
        pass


def _wav_duration_sec(wav_path: str) -> float:
    import wave

    try:
        with wave.open(wav_path, "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate() or 0
            if rate <= 0:
                return 0.0
            return float(frames) / float(rate)
    except Exception:
        return 0.0


def _load_stats() -> dict:
    try:
        with open(STATS_PATH, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _save_stats(data: dict) -> None:
    try:
        with open(STATS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _get_rtf_est(kind: str) -> float:
    stats = _load_stats()
    try:
        return float(((stats.get("rtf") or {}).get(kind) or 0.0) or 0.0)
    except Exception:
        return 0.0


def _update_rtf_est(kind: str, rtf: float) -> None:
    if not (rtf and rtf > 0):
        return
    stats = _load_stats()
    rtf_map = stats.setdefault("rtf", {})
    prev = float(rtf_map.get(kind) or 0.0) if isinstance(rtf_map.get(kind), (int, float)) else 0.0
    alpha = 0.3
    rtf_map[kind] = (prev * (1 - alpha) + rtf * alpha) if prev > 0 else rtf
    _save_stats(stats)


class _ProgressState:
    def __init__(self, *, audio_sec: float, est_rtf: float) -> None:
        self._lock = threading.Lock()
        self.audio_sec = float(audio_sec or 0.0)
        self.est_rtf = float(est_rtf or 0.0)
        self.processed_sec = 0.0
        self.est_total_sec = max(1.0, self.audio_sec * (self.est_rtf or 3.0))

    def set_processed_sec(self, sec: float) -> None:
        with self._lock:
            self.processed_sec = max(self.processed_sec, float(sec or 0.0))

    def snapshot(self) -> tuple[float, float, float]:
        with self._lock:
            return float(self.audio_sec), float(self.processed_sec), float(self.est_total_sec)

    def update_estimate(self, elapsed_sec: float) -> None:
        with self._lock:
            audio_sec = float(self.audio_sec)
            processed_sec = float(self.processed_sec)
            prev = float(self.est_total_sec)

            if audio_sec > 0 and processed_sec > 0:
                progress = min(processed_sec / audio_sec, 0.995)
                raw = max(elapsed_sec / max(1e-3, progress), elapsed_sec)
                # Smooth to avoid jumping ETA.
                self.est_total_sec = max(elapsed_sec, prev * 0.7 + raw * 0.3)
                return

            # No progress info yet (e.g., model warmup/download) — keep a conservative estimate.
            base = audio_sec * (self.est_rtf or 3.0)
            self.est_total_sec = max(elapsed_sec, prev, base)


def _parse_timeout_env(name: str) -> Optional[float]:
    """
    Returns:
    - None when unset/empty (meaning: use dynamic default)
    - 0.0 when set to 0 or negative (meaning: disable timeout)
    - float seconds when set to a positive number
    """
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return None
    try:
        v = float(raw)
    except Exception:
        return None
    if v <= 0:
        return 0.0
    return v


def _default_whisper_timeout_sec(*, audio_sec: float) -> float:
    """
    Dynamic default timeout for Whisper (seconds).
    The old fixed 240s is too small for long audio; keep a conservative scaling.
    """
    audio_sec = float(audio_sec or 0.0)
    rtf = _get_rtf_est("whisper") or 0.0
    if rtf <= 0:
        # Conservative fallback (CPU medium can be slow on some machines).
        rtf = 0.6
    est = max(60.0, audio_sec * rtf)
    # Give extra slack for warmup/IO and variability.
    return max(300.0, est * 2.0 + 120.0)


def _chunks(text: str, *, limit: int = TELEGRAM_TEXT_LIMIT - 50):
    text = text or ""
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)
        if cut <= 0:
            cut = limit
        yield text[:cut].rstrip()
        text = text[cut:].lstrip()
    if text.strip():
        yield text


def _cancel_markup(session_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton("Отменить обработку", callback_data=f"cancel:{session_id}")]]
    )


def _media_group_choice_markup(token: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("По-отдельности", callback_data=f"mediagroup:sep:{token}")],
            [InlineKeyboardButton("Склеить и как один", callback_data=f"mediagroup:merge:{token}")],
        ]
    )


class _UserCancelled(Exception):
    pass


def _active_sessions(context: ContextTypes.DEFAULT_TYPE) -> dict:
    return context.application.bot_data.setdefault("active_sessions", {})


def _asr_queue(context: ContextTypes.DEFAULT_TYPE) -> list[str]:
    return context.application.bot_data.setdefault("asr_queue", [])


def _asr_state_lock(context: ContextTypes.DEFAULT_TYPE) -> asyncio.Lock:
    return context.application.bot_data.setdefault("asr_state_lock", asyncio.Lock())


def _media_group_state_lock(context: ContextTypes.DEFAULT_TYPE) -> asyncio.Lock:
    return context.application.bot_data.setdefault("media_group_state_lock", asyncio.Lock())


def _pending_media_groups(context: ContextTypes.DEFAULT_TYPE) -> dict:
    return context.application.bot_data.setdefault("pending_media_groups", {})


def _pending_media_group_choices(context: ContextTypes.DEFAULT_TYPE) -> dict:
    return context.application.bot_data.setdefault("pending_media_group_choices", {})


def _cleanup_media_group_choices(choices: dict) -> None:
    now = time.monotonic()
    for token, payload in list(choices.items()):
        created = float((payload or {}).get("created_monotonic") or 0.0)
        if created <= 0 or (now - created) > MEDIA_GROUP_CHOICE_TTL_SEC:
            choices.pop(token, None)


async def _asr_get_state(context: ContextTypes.DEFAULT_TYPE, *, session_id: str) -> tuple[int, int, int]:
    """
    Returns (active, position, limit). Position is 1-based, 0 if not in queue.
    """
    lock = _asr_state_lock(context)
    async with lock:
        active = int(context.application.bot_data.get("asr_active", 0) or 0)
        limit = int(context.application.bot_data.get("asr_semaphore_limit", _asr_concurrency()) or _asr_concurrency())
        q = _asr_queue(context)
        try:
            pos = q.index(session_id) + 1
        except ValueError:
            pos = 0
        return active, pos, limit


async def _asr_enqueue(context: ContextTypes.DEFAULT_TYPE, *, session_id: str) -> tuple[int, int, int]:
    """
    Enqueue session_id. Returns (active, position, limit).
    """
    lock = _asr_state_lock(context)
    async with lock:
        q = _asr_queue(context)
        if session_id not in q:
            q.append(session_id)
        active = int(context.application.bot_data.get("asr_active", 0) or 0)
        limit = int(context.application.bot_data.get("asr_semaphore_limit", _asr_concurrency()) or _asr_concurrency())
        return active, q.index(session_id) + 1, limit


async def _asr_dequeue(context: ContextTypes.DEFAULT_TYPE, *, session_id: str) -> None:
    lock = _asr_state_lock(context)
    async with lock:
        q = _asr_queue(context)
        try:
            q.remove(session_id)
        except ValueError:
            pass


async def _asr_active_inc(context: ContextTypes.DEFAULT_TYPE) -> None:
    lock = _asr_state_lock(context)
    async with lock:
        cur = int(context.application.bot_data.get("asr_active", 0) or 0)
        context.application.bot_data["asr_active"] = cur + 1


async def _asr_active_dec(context: ContextTypes.DEFAULT_TYPE) -> None:
    lock = _asr_state_lock(context)
    async with lock:
        cur = int(context.application.bot_data.get("asr_active", 0) or 0)
        context.application.bot_data["asr_active"] = max(0, cur - 1)


async def _queue_wait_updater(
    context: ContextTypes.DEFAULT_TYPE,
    *,
    message,
    session_id: str,
    stop_event: asyncio.Event,
) -> None:
    try:
        while not stop_event.is_set():
            active, pos, limit = await _asr_get_state(context, session_id=session_id)
            if pos <= 0:
                return
            word = _ru_plural(active, "транскрибация", "транскрибации", "транскрибаций")
            text = f"⏳ Сейчас обрабатывается {active} {word}.\nВаш файл в очереди под номером {pos}."
            try:
                await _safe_edit_message(message, text, reply_markup=None)
            except Exception:
                return
            await asyncio.sleep(2.0)
    except asyncio.CancelledError:
        return


def _write_active_sessions_snapshot(context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Best-effort, local-only snapshot of currently running sessions.
    Used by the dashboard to show live concurrency.
    """
    if not _truthy_env("ACTIVE_SESSIONS_ENABLED", True):
        return
    path = (os.environ.get("ACTIVE_SESSIONS_PATH") or "").strip() or DEFAULT_ACTIVE_SESSIONS_PATH
    try:
        sessions = _active_sessions(context) or {}
        items = []
        for sid, sess in sessions.items():
            if not isinstance(sess, dict):
                items.append({"session_id": sid})
                continue
            items.append(
                {
                    "session_id": sid,
                    "started_at": sess.get("started_at"),
                    "user_id": sess.get("user_id"),
                    "chat_id": sess.get("chat_id"),
                }
            )
        payload = {
            "updated_at": datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).isoformat(),
            "active_sessions": items,
            "count": len(items),
        }
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        # Must never affect bot flow.
        pass


def _ru_plural(n: int, one: str, few: str, many: str) -> str:
    n = abs(int(n))
    if (n % 10) == 1 and (n % 100) != 11:
        return one
    if 2 <= (n % 10) <= 4 and not (12 <= (n % 100) <= 14):
        return few
    return many


def _active_sessions_summary_line(context: ContextTypes.DEFAULT_TYPE) -> str:
    sessions = _active_sessions(context) or {}
    files = len(sessions)
    if files <= 0:
        return ""

    user_keys = set()
    for sid, sess in sessions.items():
        if not isinstance(sess, dict):
            user_keys.add(("s", sid))
            continue
        uid = sess.get("user_id")
        if uid:
            user_keys.add(("u", int(uid)))
        else:
            user_keys.add(("s", sid))

    users = len(user_keys)
    file_word = _ru_plural(files, "файл", "файла", "файлов")
    # Genitive after "от":
    user_word = _ru_plural(users, "пользователя", "пользователей", "пользователей")
    return f"В процессе обработки {files} {file_word} от {users} {user_word}."


def _encode_audio_preview_ffmpeg(
    *,
    ffmpeg_bin: str,
    src_wav: str,
    out_dir: str,
    base_name: str = "processed_audio",
) -> str:
    """
    Best-effort: encodes wav to a compact format for sending back to user.
    Tries MP3 first, then OGG/Opus.
    Returns output file path.
    """
    ffmpeg_bin = (ffmpeg_bin or "ffmpeg").strip() or "ffmpeg"
    os.makedirs(out_dir, exist_ok=True)

    mp3_path = os.path.join(out_dir, f"{base_name}.mp3")
    cmd_mp3 = [
        ffmpeg_bin,
        "-y",
        "-i",
        src_wav,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "libmp3lame",
        "-b:a",
        "64k",
        mp3_path,
    ]
    r = subprocess.run(cmd_mp3, capture_output=True, text=True)
    if r.returncode == 0 and os.path.exists(mp3_path):
        return mp3_path

    ogg_path = os.path.join(out_dir, f"{base_name}.ogg")
    cmd_ogg = [
        ffmpeg_bin,
        "-y",
        "-i",
        src_wav,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "libopus",
        "-b:a",
        "32k",
        "-vbr",
        "on",
        "-compression_level",
        "10",
        ogg_path,
    ]
    r2 = subprocess.run(cmd_ogg, capture_output=True, text=True)
    if r2.returncode == 0 and os.path.exists(ogg_path):
        return ogg_path

    raise RuntimeError(
        "Не удалось подготовить обработанное аудио для отправки.\n"
        f"MP3 error: {(r.stderr or '').strip()}\n"
        f"OGG error: {(r2.stderr or '').strip()}"
    )


async def _gigaam_transcribe_hard_cancel(
    *,
    wav_path: str,
    model_name: str,
    device: str,
    hf_token: Optional[str],
    cancel_event: asyncio.Event,
) -> tuple[list, dict]:
    """
    Runs GigaAM in a separate process and terminates it if cancel_event is set.
    Returns (segments, info) where segments is a list of ASRSegment-like tuples.
    """
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue(maxsize=1)
    proc = ctx.Process(
        target=gigaam_transcribe_worker,
        kwargs={
            "wav_path": wav_path,
            "model_name": model_name,
            "device": device,
            "hf_token": hf_token,
            "out_queue": q,
        },
        daemon=True,
    )
    proc.start()

    async def _wait_result():
        return await asyncio.to_thread(q.get)

    wait_task = asyncio.create_task(_wait_result())
    cancel_task = asyncio.create_task(cancel_event.wait())
    try:
        done, pending = await asyncio.wait({wait_task, cancel_task}, return_when=asyncio.FIRST_COMPLETED)
        if cancel_task in done:
            try:
                if proc.is_alive():
                    proc.terminate()
            except Exception:
                pass
            try:
                await asyncio.to_thread(proc.join, 2.0)
            except Exception:
                pass
            raise _UserCancelled()

        payload = wait_task.result()
        try:
            await asyncio.to_thread(proc.join, 2.0)
        except Exception:
            pass

        if not isinstance(payload, dict):
            raise RuntimeError("GigaAM worker returned invalid payload.")
        if not payload.get("ok"):
            err_type = str(payload.get("error_type") or "GigaAMError")
            err = str(payload.get("error") or "").strip() or err_type
            raise RuntimeError(err)

        segs = payload.get("segments") or []
        info = payload.get("info") or {}
        return segs, info
    finally:
        for t in (wait_task, cancel_task):
            if not t.done():
                t.cancel()
        try:
            if proc.is_alive():
                proc.terminate()
        except Exception:
            pass


async def cb_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    q = getattr(update, "callback_query", None)
    if not q or not getattr(q, "data", None):
        return
    data = str(q.data)
    if not data.startswith("cancel:"):
        return

    session_id = data.split(":", 1)[1].strip()
    sess = (_active_sessions(context) or {}).get(session_id)
    if not sess:
        try:
            await q.answer("Сессия уже завершена.", show_alert=False)
        except Exception:
            pass
        return

    ev = sess.get("cancel_event")
    if isinstance(ev, asyncio.Event):
        ev.set()

    try:
        await q.answer("Ок, отменяю…", show_alert=False)
    except Exception:
        pass

    # Best-effort update the status message itself (button will be removed).
    try:
        await q.message.edit_text("⛔️ Отмена запрошена — завершаю текущий шаг…", reply_markup=None)
    except Exception:
        pass


def _sanitize_filename(name: Optional[str], *, fallback: str) -> str:
    raw = os.path.basename((name or "").strip())
    if not raw:
        raw = fallback
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._")
    return safe or fallback


def _concat_wav_files(wav_paths: list[str], out_path: str) -> None:
    if len(wav_paths) < 2:
        raise ValueError("Need at least two WAV files to concatenate.")

    first_format = None
    with wave.open(out_path, "wb") as w_out:
        for p in wav_paths:
            with wave.open(p, "rb") as w_in:
                fmt = (
                    int(w_in.getnchannels()),
                    int(w_in.getsampwidth()),
                    int(w_in.getframerate()),
                    str(w_in.getcomptype()),
                    str(w_in.getcompname()),
                )
                if first_format is None:
                    first_format = fmt
                    w_out.setnchannels(fmt[0])
                    w_out.setsampwidth(fmt[1])
                    w_out.setframerate(fmt[2])
                elif fmt != first_format:
                    raise RuntimeError("WAV formats mismatch; cannot concatenate.")
                w_out.writeframes(w_in.readframes(w_in.getnframes()))


async def _run_items_separately(
    context: ContextTypes.DEFAULT_TYPE,
    *,
    items: list[dict],
    announce_text: Optional[str] = None,
) -> None:
    if not items:
        return
    ordered = sorted(items, key=lambda x: int(x.get("source_message_id") or 0))
    first_reply = ordered[0].get("reply_target")
    if announce_text and first_reply is not None:
        try:
            await first_reply.reply_text(announce_text)
        except Exception:
            pass

    total = len(ordered)
    for idx, item in enumerate(ordered, start=1):
        reply_target = item.get("reply_target") or getattr(item.get("update"), "effective_message", None)
        if reply_target is None:
            continue
        if total > 1 and reply_target is not None:
            try:
                await reply_target.reply_text(f"▶️ Файл {idx}/{total}: запускаю обработку…")
            except Exception:
                pass
        await _process_audio(
            item["update"],
            context,
            file_id=str(item["file_id"]),
            filename=str(item.get("filename") or "audio"),
            reply_target=reply_target,
            source_message_id=item.get("source_message_id"),
            source_file_size=item.get("source_file_size"),
            normalize_audio=bool(item.get("normalize_audio")),
        )


async def _run_items_merged(context: ContextTypes.DEFAULT_TYPE, *, items: list[dict]) -> None:
    if len(items) < 2:
        await _run_items_separately(context, items=items)
        return

    ordered = sorted(items, key=lambda x: int(x.get("source_message_id") or 0))
    first = ordered[0]
    reply_target = first.get("reply_target") or getattr(first.get("update"), "effective_message", None)
    if reply_target is None:
        await _run_items_separately(context, items=ordered)
        return
    total = len(ordered)
    file_word = _ru_plural(total, "файл", "файла", "файлов")
    status = None
    try:
        if reply_target is not None:
            status = await reply_target.reply_text(f"📎 Склеиваю {total} {file_word} в один…")

        with tempfile.TemporaryDirectory(prefix="tg_asr_merge2_") as td:
            ap = AudioProcessor()
            loop = asyncio.get_running_loop()
            wav_parts: list[str] = []
            total_size = 0

            for idx, item in enumerate(ordered, start=1):
                if status is not None:
                    await _safe_edit(status, f"📥 Скачиваю файл {idx}/{total}…")
                tg_file = await context.bot.get_file(str(item["file_id"]))
                safe_name = _sanitize_filename(item.get("filename"), fallback=f"part{idx}.bin")
                src_path = os.path.join(td, f"{idx}_{safe_name}")
                await tg_file.download_to_drive(custom_path=src_path)
                total_size += int(item.get("source_file_size") or 0)
                if status is not None:
                    await _safe_edit(status, f"🎛 Готовлю WAV для файла {idx}/{total}…")
                wav_path = await loop.run_in_executor(None, lambda p=src_path: ap.extract_audio(p, output_dir=td))
                wav_parts.append(wav_path)

            merged_input = os.path.join(td, "merged_input.wav")
            if status is not None:
                await _safe_edit(status, "📎 Склеиваю WAV…")
            await loop.run_in_executor(None, lambda: _concat_wav_files(wav_parts, merged_input))
            await _safe_delete(status)

            await _process_audio(
                first["update"],
                context,
                file_id=f"merged:{total}:{uuid.uuid4().hex[:8]}",
                filename="merged_input.wav",
                reply_target=reply_target,
                source_message_id=first.get("source_message_id"),
                source_file_size=(total_size or None),
                normalize_audio=any(bool(item.get("normalize_audio")) for item in ordered),
                local_source_path=merged_input,
                working_dir=td,
            )
    except Exception as exc:
        await _safe_delete(status)
        if reply_target is not None:
            await reply_target.reply_text(f"Ошибка при склейке файлов: {str(exc).strip() or exc.__class__.__name__}")


async def _offer_media_group_choice(context: ContextTypes.DEFAULT_TYPE, *, items: list[dict]) -> None:
    if len(items) < 2:
        await _run_items_separately(context, items=items)
        return

    ordered = sorted(items, key=lambda x: int(x.get("source_message_id") or 0))
    first = ordered[0]
    reply_target = first.get("reply_target") or getattr(first.get("update"), "effective_message", None)
    if reply_target is None:
        await _run_items_separately(context, items=ordered)
        return

    total = len(ordered)
    file_word = _ru_plural(total, "файл", "файла", "файлов")
    token = secrets.token_urlsafe(8)
    try:
        prompt = await reply_target.reply_text(
            f"Получил {total} {file_word} одновременно.\n"
            "Как обработать?\n"
            "• Вместе: склеить в одну длинную запись и транскрибировать как один файл.\n"
            "• По-отдельности: обработать каждый файл отдельно.",
            reply_markup=_media_group_choice_markup(token),
        )
    except Exception:
        await _run_items_separately(context, items=ordered)
        return

    lock = _media_group_state_lock(context)
    async with lock:
        choices = _pending_media_group_choices(context)
        _cleanup_media_group_choices(choices)
        choices[token] = {
            "items": ordered,
            "chat_id": int(getattr(getattr(first.get("update"), "effective_chat", None), "id", 0) or 0),
            "user_id": int(getattr(getattr(first.get("update"), "effective_user", None), "id", 0) or 0),
            "prompt_message_id": int(getattr(prompt, "message_id", 0) or 0),
            "created_monotonic": time.monotonic(),
        }


async def _finalize_media_group(context: ContextTypes.DEFAULT_TYPE, *, key: str) -> None:
    try:
        await asyncio.sleep(MEDIA_GROUP_WAIT_SEC)
        lock = _media_group_state_lock(context)
        async with lock:
            pending = _pending_media_groups(context)
            group = pending.pop(key, None)
        if not group:
            return

        items = list(group.get("items") or [])
        if len(items) >= 2:
            await _offer_media_group_choice(context, items=items)
            return

        if len(items) >= 1:
            await _run_items_separately(context, items=items)
    except Exception:
        logging.getLogger("local_telegram_bot").exception("Failed to finalize media-group files")


async def _handle_media_group_file(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    *,
    file_id: str,
    filename: str,
    file_size: Optional[int],
) -> bool:
    msg = update.effective_message
    if msg is None:
        return False
    media_group_id = getattr(msg, "media_group_id", None)
    if not media_group_id:
        return False
    chat_id = int(getattr(getattr(update, "effective_chat", None), "id", 0) or 0)
    if chat_id == 0:
        return False

    item = {
        "update": update,
        "file_id": file_id,
        "filename": filename or "audio",
        "source_file_size": file_size,
        "source_message_id": int(getattr(msg, "message_id", 0) or 0) or None,
        "reply_target": msg,
        "normalize_audio": False,
    }
    key = f"{chat_id}:{media_group_id}"

    should_start_task = False
    lock = _media_group_state_lock(context)
    async with lock:
        pending = _pending_media_groups(context)
        group = pending.get(key)
        if group is None:
            group = {
                "chat_id": chat_id,
                "media_group_id": str(media_group_id),
                "items": [],
                "created_monotonic": time.monotonic(),
                "task": None,
            }
            pending[key] = group

        msg_id = int(item.get("source_message_id") or 0)
        exists = any(int((it or {}).get("source_message_id") or 0) == msg_id for it in (group.get("items") or []))
        if not exists:
            group["items"].append(item)
        should_start_task = group.get("task") is None

    if should_start_task:
        t = asyncio.create_task(_finalize_media_group(context, key=key))
        lock = _media_group_state_lock(context)
        async with lock:
            group = _pending_media_groups(context).get(key)
            if group is not None and group.get("task") is None:
                group["task"] = t
    return True


async def cb_media_group_choice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    q = getattr(update, "callback_query", None)
    if not q or not getattr(q, "data", None):
        return
    data = str(q.data)
    if not (data.startswith("mediagroup:") or data.startswith("twofiles:")):
        return

    parts = data.split(":", 2)
    if len(parts) != 3:
        return
    mode, token = parts[1].strip(), parts[2].strip()
    if mode not in {"sep", "merge"} or not token:
        return

    current_user_id = int(getattr(getattr(update, "effective_user", None), "id", 0) or 0)
    payload = None
    lock = _media_group_state_lock(context)
    async with lock:
        choices = _pending_media_group_choices(context)
        _cleanup_media_group_choices(choices)
        candidate = choices.get(token)
        expected_user_id = int((candidate or {}).get("user_id") or 0)
        if not candidate:
            payload = None
        elif expected_user_id and current_user_id and expected_user_id != current_user_id:
            payload = "__forbidden__"
        else:
            payload = choices.pop(token, None)

    if payload is None:
        try:
            await q.answer("Выбор уже неактуален.", show_alert=False)
        except Exception:
            pass
        return
    if payload == "__forbidden__":
        try:
            await q.answer("Эта кнопка не для вас.", show_alert=True)
        except Exception:
            pass
        return

    try:
        await q.answer("Принято.", show_alert=False)
    except Exception:
        pass
    try:
        items = list((payload or {}).get("items") or [])
        total = len(items)
        file_word = _ru_plural(total, "файл", "файла", "файлов")
        if mode == "merge":
            await q.message.edit_text(
                f"Выбрано: склеить {total} {file_word} и обработать как один.",
                reply_markup=None,
            )
        else:
            await q.message.edit_text(
                f"Выбрано: обработать {total} {file_word} по-отдельности.",
                reply_markup=None,
            )
    except Exception:
        pass

    if mode == "merge":
        await _run_items_merged(context, items=items)
    else:
        await _run_items_separately(context, items=items)


async def _reply_long(update: Update, text: str) -> None:
    for part in _chunks(text):
        await update.effective_message.reply_text(part)


async def _safe_edit(message, text: str) -> Optional[float]:
    try:
        if not message:
            return None
        await message.edit_text(text)
        return None
    except RetryAfter as exc:
        return float(getattr(exc, "retry_after", 0) or 0) or 1.0
    except Exception:
        # Message can be deleted/edited too often/etc. Don't fail ASR because of UI.
        return None


async def _safe_edit_formatted(message, text: str, *, parse_mode: Optional[str] = None) -> Optional[float]:
    """
    Like `_safe_edit`, but allows setting `parse_mode` (HTML/MarkdownV2).
    """
    try:
        if not message:
            return None
        await message.edit_text(text, parse_mode=parse_mode)
        return None
    except RetryAfter as exc:
        return float(getattr(exc, "retry_after", 0) or 0) or 1.0
    except Exception:
        return None


async def _safe_edit_message(
    message,
    text: str,
    *,
    parse_mode: Optional[str] = None,
    reply_markup=None,
) -> Optional[float]:
    """
    Like `_safe_edit`, but supports parse_mode and reply_markup (inline keyboards).
    """
    try:
        if not message:
            return None
        await message.edit_text(text, parse_mode=parse_mode, reply_markup=reply_markup)
        return None
    except RetryAfter as exc:
        return float(getattr(exc, "retry_after", 0) or 0) or 1.0
    except Exception:
        return None


async def _safe_delete(message) -> None:
    try:
        if message:
            await message.delete()
    except Exception:
        pass


async def _ticker(
    *,
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    message,
    base_text: str,
    reply_markup=None,
    state: Optional[_ProgressState] = None,
    cancel_event: Optional[asyncio.Event] = None,
    interval_sec: float = 5.0,
) -> None:
    start = time.monotonic()
    try:
        while True:
            if cancel_event is not None and cancel_event.is_set():
                return
            elapsed_sec = time.monotonic() - start
            if state is not None:
                state.update_estimate(elapsed_sec)
                audio_sec, processed_sec, est_total_sec = state.snapshot()
                if audio_sec > 0 and processed_sec > 0:
                    pct = int(min(99, (processed_sec / audio_sec) * 100))
                    line = f"⏱ {_fmt_dur(elapsed_sec)}/{_fmt_dur(est_total_sec)} • {pct}%"
                else:
                    line = f"⏱ {_fmt_dur(elapsed_sec)}/{_fmt_dur(est_total_sec)}"
            else:
                line = f"⏱ {_fmt_dur(elapsed_sec)}"

            summary = _active_sessions_summary_line(context)
            text = f"{base_text}\n{line}"
            if summary:
                text += f"\n{summary}"
            retry_after = await _safe_edit_message(message, text, reply_markup=reply_markup)
            try:
                await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
            except RetryAfter as exc:
                retry_after = max(retry_after or 0.0, float(getattr(exc, "retry_after", 0) or 0) or 1.0)
            except Exception:
                pass

            sleep_for = interval_sec
            if retry_after is not None and retry_after > 0:
                sleep_for = max(sleep_for, retry_after + 0.5)
            await asyncio.sleep(sleep_for)
    except asyncio.CancelledError:
        return


def _get_env(name: str, default: str) -> str:
    v = (os.environ.get(name) or "").strip()
    return v if v else default


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _ = context
    domain = _auth_domain()
    intro = (
        "Добро пожаловать в идеальный (ну почти :)) транскрибатор русских аудио сообщений. "
        "Я использую для транскрибации две SOTA модели: GigaAM и Whisper (medium), "
        "GigaAM - лучше русский, а Whisper - хорошо распознает смешанный ru/en текст. "
        "Затем с помощью LLM делаю их сравнительный анализ и итоговый вариант транскрибации. "
        "LLM правит смысловые ошибки, пунктуацию, делает разбивку на абзацы и предложения.\n\n"
        "Бот хорошо подходит для задач, когда важна точность транскрибации, а не скорость: "        
        "подготовка постов для social media, должностных инструкций и т.п.\n\n"
    )

    privacy_block = (
        "Политика конфиденциальности:\n"
        "- Бот не сохраняет текст транскрибации и LLM-промпты; в usage-лог попадают только метаданные (username/ID пользователя и ID чата).\n"
        "- Для внешних пользователей используется локальная LLM на моем компьютере и ваши аудио и тексты нигде не сохраняются.\n"
        "- Запросы сотрудников Business Booster обрабатывает Gemini 3 Pro в нашем корпоративном аккаунте.\n\n"
    )

    source_block = (
        "Исходный код:\n"
        "https://github.com/vlntnbb/ideal-russian-transcriber\n"
        "Каждый может поднять такого бота у себя и доработать как нужно.\n"
    )

    if _auth_enabled():
        auth_block = (
            "Если вы сотрудник Business Booster, авторизуйтесь по корпоративной почте, "
            "тогда вам будет доступна более качественная платная модель Gemini 3 Pro.\n\n"
            f"Просто отправьте мне вашу почту вида `email@{domain}`.\n\n"
            "Если вы гость, то можете использовать бот бесплатно на локальной модели "
            "(это немного медленнее и чуть менее точно, но тоже неплохо)."
        )
    else:
        auth_block = (
            "Авторизация отключена (AUTH_ENABLED=0). "
            "Бот будет работать с доступной LLM без проверки домена."
        )

    await update.effective_message.reply_text(intro + privacy_block + auth_block + "\n\n" + source_block)


async def cmd_auth(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.effective_message
    user = update.effective_user
    if msg is None or user is None:
        return

    user_id = int(user.id)
    if _auth_is_user_authorized(user_id):
        await msg.reply_text("✅ Вы уже авторизованы.")
        return

    if not _auth_enabled():
        await msg.reply_text("Авторизация отключена (AUTH_ENABLED=0). Ничего делать не нужно.")
        return

    domain = _auth_domain()
    email = (" ".join(context.args or [])).strip()
    if not email:
        await msg.reply_text(f"Отправьте мне вашу почту вида `email@{domain}`.")
        return

    email = email.strip().lower()
    # Minimal email validation (enough for domain-gated auth).
    # NOTE: must use `\s` (whitespace), not `\\s` (literal "s" + backslash in a char-class).
    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        await msg.reply_text(f"Неверный email. Пример: name@{domain}")
        return
    if not email.endswith(f"@{domain}"):
        await msg.reply_text(f"Авторизация доступна только для домена {domain}.")
        return

    code = _auth_start_pending(user_id, email)
    ttl_min = max(1, int(_auth_code_ttl_sec() / 60))

    subject = "Ideal Russian Transcriber — код авторизации"
    body = (
        "Чтобы авторизоваться в боте, отправьте в Telegram следующую фразу (слова и порядок важны):\n\n"
        f"{code}\n\n"
        f"Код действителен {ttl_min} минут."
    )

    await msg.reply_text(f"📧 Отправляю код на {email}…")
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(None, lambda: _smtp_send_email(to_email=email, subject=subject, body=body))
    except Exception as exc:
        logging.getLogger("local_telegram_bot").exception("Failed to send auth email")
        await msg.reply_text(
            "Не удалось отправить письмо. Проверьте SMTP настройки в `.env` "
            "(SMTP_HOST/SMTP_PORT/SMTP_USERNAME/SMTP_PASSWORD/SMTP_FROM).\n"
            f"Ошибка: {exc}"
        )
        return

    mn, mx = _auth_code_word_count()
    await msg.reply_text(f"✅ Письмо отправлено на {email}.\nПришлите сюда код из {mn}-{mx} слов (как в письме).")


async def cmd_llm(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.effective_message
    user = update.effective_user
    if msg is None or user is None:
        return

    if not _auth_enabled():
        await msg.reply_text("Команда доступна только при включенной авторизации (`AUTH_ENABLED=1`).")
        return

    user_id = int(user.id)
    if not _auth_is_user_authorized(user_id):
        domain = _auth_domain()
        await msg.reply_text(
            "Команда доступна только авторизованным пользователям.\n"
            f"Сначала авторизуйтесь: отправьте почту вида `email@{domain}`."
        )
        return

    arg = ((context.args or [""])[0] or "").strip().lower()
    if not arg or arg in {"status", "show"}:
        current = _auth_get_user_llm_provider(user_id)
        current_label = "Gemini" if current == LLM_PROVIDER_GEMINI else "Local (Ollama)"
        gemini_ready = bool((os.environ.get("GEMINI_API_KEY") or "").strip())
        await msg.reply_text(
            f"Текущая LLM: {current_label}.\n"
            "Смена: `/llm gemini` или `/llm local`.\n"
            f"GEMINI_API_KEY: {'задан' if gemini_ready else 'не задан (при выборе Gemini будет fallback на Local)'}."
        )
        return

    target = _normalize_llm_provider(arg)
    if target is None:
        await msg.reply_text("Неизвестный вариант. Используйте: `/llm gemini` или `/llm local`.")
        return

    if not _auth_set_user_llm_provider(user_id, target):
        await msg.reply_text("Не удалось сохранить выбор LLM. Попробуйте ещё раз.")
        return

    if target == LLM_PROVIDER_GEMINI:
        gemini_ready = bool((os.environ.get("GEMINI_API_KEY") or "").strip())
        if gemini_ready:
            await msg.reply_text("✅ Выбрана LLM: Gemini.")
        else:
            await msg.reply_text("✅ Выбрана LLM: Gemini.\n⚠️ GEMINI_API_KEY не задан, поэтому фактически будет использована Local (Ollama).")
        return

    await msg.reply_text("✅ Выбрана LLM: Local (Ollama).")


async def _begin_email_auth(update: Update, context: ContextTypes.DEFAULT_TYPE, *, email: str) -> None:
    """
    Starts email authorization flow from a plain email message (no /auth command required).
    """
    msg = update.effective_message
    user = update.effective_user
    if msg is None or user is None:
        return

    if not _auth_enabled():
        await msg.reply_text("Авторизация отключена (AUTH_ENABLED=0).")
        return

    user_id = int(user.id)
    domain = _auth_domain()
    email = (email or "").strip().lower()

    if _auth_is_user_authorized(user_id):
        await msg.reply_text("✅ Вы уже авторизованы.")
        return

    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        await msg.reply_text(f"Неверный email. Пример: name@{domain}")
        return
    if not email.endswith(f"@{domain}"):
        await msg.reply_text(f"Авторизация доступна только для домена {domain}.")
        return

    code = _auth_start_pending(user_id, email)
    ttl_min = max(1, int(_auth_code_ttl_sec() / 60))

    subject = "Ideal Russian Transcriber — код авторизации"
    body = (
        "Чтобы авторизоваться в боте, отправьте в Telegram следующую фразу (слова и порядок важны):\n\n"
        f"{code}\n\n"
        f"Код действителен {ttl_min} минут."
    )

    await msg.reply_text(f"📧 Отправляю код на {email}…")
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(None, lambda: _smtp_send_email(to_email=email, subject=subject, body=body))
    except Exception as exc:
        logging.getLogger("local_telegram_bot").exception("Failed to send auth email")
        await msg.reply_text(
            "Не удалось отправить письмо. Проверьте SMTP настройки в `.env` "
            "(SMTP_HOST/SMTP_PORT/SMTP_USERNAME/SMTP_PASSWORD/SMTP_FROM).\n"
            f"Ошибка: {exc}"
        )
        return

    mn, mx = _auth_code_word_count()
    await msg.reply_text(f"✅ Письмо отправлено на {email}.\nПришлите сюда код из {mn}-{mx} слов (как в письме).")


def _pick_telegram_file(update: Update):
    msg = update.effective_message
    if msg is None:
        return None, None, None
    return _pick_telegram_file_from_message(msg)


def _pick_telegram_file_from_message(msg):
    if msg.voice:
        return msg.voice.file_id, "voice.ogg", msg.voice.file_size
    if msg.audio:
        name = msg.audio.file_name or "audio"
        return msg.audio.file_id, name, msg.audio.file_size
    if msg.video:
        # Telegram Video doesn't always carry file_name.
        ext = "mp4"
        try:
            mt = (msg.video.mime_type or "").lower()
            if "webm" in mt:
                ext = "webm"
            elif "quicktime" in mt:
                ext = "mov"
            elif "mp4" in mt:
                ext = "mp4"
        except Exception:
            pass
        return msg.video.file_id, f"video.{ext}", msg.video.file_size
    if getattr(msg, "video_note", None):
        # video_note is typically an mp4 without filename.
        return msg.video_note.file_id, "video_note.mp4", msg.video_note.file_size
    if msg.document and (msg.document.mime_type or "").startswith("audio/"):
        name = msg.document.file_name or "audio.bin"
        return msg.document.file_id, name, msg.document.file_size
    if msg.document and (msg.document.mime_type or "").startswith("video/"):
        name = msg.document.file_name or "video.bin"
        return msg.document.file_id, name, msg.document.file_size

    return None, None, None


def _telegram_max_get_file_bytes() -> int:
    """
    Telegram Bot API download limit precheck (in MB).
    Set TELEGRAM_MAX_GET_FILE_MB=0 to disable precheck.
    """
    raw = (os.environ.get("TELEGRAM_MAX_GET_FILE_MB") or "").strip()
    mb = int(raw or "20")
    if mb <= 0:
        return 0
    return mb * 1024 * 1024


def _file_too_big_message(*, size_mb: float, limit_mb: float) -> str:
    """
    User-facing guidance for speech audio compression.
    """
    size_mb = max(0.0, float(size_mb))
    limit_mb = max(0.0, float(limit_mb)) or 20.0

    # Rough estimates: MB/min ≈ (kbps * 60) / 8 / 1024
    mp3_kbps = 48
    ogg_kbps = 24
    mp3_mb_per_min = (mp3_kbps * 60) / 8 / 1024
    ogg_mb_per_min = (ogg_kbps * 60) / 8 / 1024
    mp3_max_min = int(limit_mb / mp3_mb_per_min) if mp3_mb_per_min > 0 else 0
    ogg_max_min = int(limit_mb / ogg_mb_per_min) if ogg_mb_per_min > 0 else 0

    return (
        f"Ошибка: файл слишком большой для скачивания ботом через Telegram API "
        f"({size_mb:.1f}MB, лимит ~{limit_mb:.0f}MB). "
        f"Сожмите файл, чтобы он удовлетворял этому ограничению и отправьте снова.\n\n"
        f"Как сжать голос (ffmpeg):\n"
        f"1) MP3 (хорошее качество речи):\n"
        f"   ffmpeg -i input.mp3 -vn -ac 1 -ar 24000 -b:a {mp3_kbps}k output.mp3\n"
        f"   ~{mp3_mb_per_min:.2f}MB/мин (≈ до {mp3_max_min} мин в {limit_mb:.0f}MB)\n"
        f"2) OGG/Opus (обычно меньше, подходит для voice):\n"
        f"   ffmpeg -i input.mp3 -vn -c:a libopus -application voip -vbr on -ac 1 -b:a {ogg_kbps}k output.ogg\n"
        f"   ~{ogg_mb_per_min:.2f}MB/мин (≈ до {ogg_max_min} мин в {limit_mb:.0f}MB)\n"
    )


async def _send_transcript_file(update: Update, *, whisper_text: str, gigaam_text: str) -> None:
    msg = update.effective_message
    if msg is None:
        return

    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    chat_id = update.effective_chat.id if update.effective_chat else "chat"
    filename = f"transcript_{chat_id}_{now}.txt"

    content = (
        "Whisper:\n"
        f"{(whisper_text or '').strip()}\n\n"
        "GigaAM:\n"
        f"{(gigaam_text or '').strip()}\n"
    )
    data = content.encode("utf-8")
    bio = io.BytesIO(data)
    bio.name = filename
    await msg.reply_document(document=bio, filename=filename)


async def _send_markdown_file(
    update: Update,
    *,
    final_text: str,
    final_error: str,
    final_label: str,
    whisper_text: str,
    gigaam_text: str,
) -> None:
    msg = update.effective_message
    if msg is None:
        return

    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    chat_id = update.effective_chat.id if update.effective_chat else "chat"
    filename = f"transcript_{chat_id}_{now}.md"

    content = render_markdown(
        final_text=final_text,
        final_error=final_error,
        final_label=final_label,
        whisper_text=whisper_text,
        gigaam_text=gigaam_text,
    )
    bio = io.BytesIO(content.encode("utf-8"))
    bio.name = filename
    await msg.reply_document(document=bio, filename=filename)


def _parse_float_env(value: str, default: float) -> float:
    try:
        return float((value or "").strip())
    except Exception:
        return float(default)


def _parse_int_env(value: str, default: int) -> int:
    try:
        return int((value or "").strip())
    except Exception:
        return int(default)


def _thinking_level_value(level: str) -> str:
    v = (level or "").strip().lower()
    # Gemini API expects plain strings ("high"/"low"/"THINKING_LEVEL_UNSPECIFIED") for v1beta.
    if v in {"high", "h"}:
        return "high"
    if v in {"medium", "mid", "m"}:
        return "medium"
    if v in {"low", "l"}:
        return "low"
    return "THINKING_LEVEL_UNSPECIFIED"


def _media_resolution_value(value: str) -> str:
    v = (value or "").strip().lower()
    # Gemini API accepts enum-like strings for mediaResolution; keep mapping for future multimodal use.
    if v in {"default", "unspecified", ""}:
        return "MEDIA_RESOLUTION_UNSPECIFIED"
    if v in {"low"}:
        return "MEDIA_RESOLUTION_LOW"
    if v in {"medium", "mid"}:
        return "MEDIA_RESOLUTION_MEDIUM"
    if v in {"high"}:
        return "MEDIA_RESOLUTION_HIGH"
    return "MEDIA_RESOLUTION_UNSPECIFIED"


async def _gemini_generate(
    *,
    api_key: str,
    model: str,
    prompt: str,
    system_prompt: Optional[str],
    generation_config: dict,
) -> str:
    url = f"{GEMINI_API_URL}/models/{model}:generateContent"
    params = {"key": api_key}
    payload = {
        "systemInstruction": {"role": "system", "parts": [{"text": system_prompt}]} if system_prompt else None,
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": generation_config,
    }
    payload = {k: v for k, v in payload.items() if v is not None}

    timeout = httpx.Timeout(connect=10.0, read=240.0, write=30.0, pool=30.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(url, params=params, json=payload)
        r.raise_for_status()
        data = r.json()

    candidates = data.get("candidates") or []
    if not candidates:
        raise RuntimeError(f"Gemini: empty response: {data!r}")
    content = (candidates[0] or {}).get("content") or {}
    parts = content.get("parts") or []
    texts = []
    for p in parts:
        t = (p or {}).get("text")
        if t:
            texts.append(str(t))
    out = "\n".join(texts).strip()
    return out


def _trim_for_telegram(text: str, *, limit: int = TELEGRAM_TEXT_LIMIT - 50) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    # Keep the tail (usually most relevant in a live "thoughts" stream).
    return "…\n" + text[-limit:].lstrip()


def _markdown_bold_lines_to_html(text: str) -> str:
    """
    Minimal Markdown -> HTML conversion for lines like "**Header**" so Telegram shows them bold.
    Everything else is HTML-escaped (no other markdown is interpreted).
    """
    out_lines = []
    for line in (text or "").splitlines():
        s = line.strip()
        if s.startswith("**") and s.endswith("**") and len(s) >= 4 and s.count("**") == 2:
            inner = s[2:-2].strip()
            out_lines.append(f"<b>{_html.escape(inner)}</b>")
        else:
            out_lines.append(_html.escape(line))
    return "\n".join(out_lines).strip()


def _markdown_to_telegram_html(text: str) -> str:
    """
    Minimal Markdown -> Telegram HTML:
    - Headings like '### Title' become bold lines.
    - Inline **bold** becomes <b>bold</b> (within a single line).
    Everything else is HTML-escaped.
    """

    def md_line_to_html(line: str) -> str:
        raw = line.rstrip("\n")
        stripped = raw.strip()

        # Headings: "#", "##", ... "######"
        m = re.match(r"^(#{1,6})\\s+(.*)$", stripped)
        if m:
            title = m.group(2).strip()
            return f"<b>{_html.escape(title)}</b>"

        # Inline bold: **...** (same line only)
        parts = raw.split("**")
        # Even length => unmatched ** markers; keep as plain text.
        if len(parts) >= 3 and (len(parts) % 2 == 1):
            out = []
            for i, part in enumerate(parts):
                if i % 2 == 1:
                    out.append(f"<b>{_html.escape(part)}</b>")
                else:
                    out.append(_html.escape(part))
            return "".join(out)

        return _html.escape(raw)

    lines = (text or "").splitlines()
    return "\n".join(md_line_to_html(line) for line in lines).strip()


def _split_html_for_telegram(html_text: str, *, limit: int = TELEGRAM_TEXT_LIMIT - 50):
    """
    Splits HTML text into chunks that Telegram can accept, trying to avoid
    cutting inside tags and keeping <b> tags balanced across chunks.
    """
    s = (html_text or "").strip()
    if not s:
        return

    open_bold = False
    while s:
        if len(s) <= limit:
            chunk = s
            s = ""
        else:
            # Prefer cutting at a newline or space before the limit.
            cut = max(s.rfind("\n", 0, limit), s.rfind(" ", 0, limit))
            if cut <= 0:
                cut = limit

            # Avoid cutting inside an HTML tag.
            sub = s[:cut]
            lt = sub.rfind("<")
            gt = sub.rfind(">")
            if lt > gt:
                # We're inside a tag; back up to before the tag.
                cut = lt
                if cut <= 0:
                    cut = limit

            chunk = s[:cut].rstrip()
            s = s[cut:].lstrip()

        # Track bold tag state and balance within the chunk.
        opens = chunk.count("<b>")
        closes = chunk.count("</b>")
        if open_bold:
            chunk = "<b>" + chunk
            open_bold = False
            opens += 1
        if opens > closes:
            open_bold = True
            chunk = chunk + "</b>"

        if chunk.strip():
            yield chunk


async def _reply_long_html(update: Update, html_text: str) -> None:
    for part in _split_html_for_telegram(html_text):
        await update.effective_message.reply_text(part, parse_mode=ParseMode.HTML)


def _gemini_chat_excerpt(text: str) -> str:
    """
    For chat output, show only the final text section (before notes/actions).
    Full output always goes to the markdown file.
    """
    s = (text or "").strip()
    if not s:
        return ""
    markers = [
        "### Примечания для контент-менеджера",
        "### Примечания",
    ]
    for m in markers:
        idx = s.find(m)
        if idx != -1:
            return s[:idx].rstrip()
    return s


async def _gemini_stream_generate(
    *,
    api_key: str,
    model: str,
    prompt: str,
    system_prompt: Optional[str],
    generation_config: dict,
    on_update,
    cancel_event: Optional[asyncio.Event] = None,
) -> str:
    """
    Streams `:streamGenerateContent` and calls `on_update(thoughts, elapsed_sec)` periodically.
    Returns final (non-thought) text.
    """
    url = f"{GEMINI_API_URL}/models/{model}:streamGenerateContent"
    # Gemini supports SSE streaming via `alt=sse`.
    params = {"key": api_key, "alt": "sse"}
    payload = {
        "systemInstruction": {"role": "system", "parts": [{"text": system_prompt}]} if system_prompt else None,
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": generation_config,
    }
    payload = {k: v for k, v in payload.items() if v is not None}

    timeout = httpx.Timeout(connect=10.0, read=600.0, write=30.0, pool=30.0)
    thoughts = ""
    final = ""
    finish_reason: Optional[str] = None
    started = time.monotonic()
    last_ui = 0.0
    ui_interval = 3.0

    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream(
            "POST",
            url,
            params=params,
            json=payload,
            headers={"Accept": "text/event-stream"},
        ) as r:
            r.raise_for_status()
            async for raw_line in r.aiter_lines():
                if cancel_event is not None and cancel_event.is_set():
                    raise _UserCancelled()
                line = (raw_line or "").strip()
                if not line:
                    continue
                if line.startswith("data:"):
                    line = line[len("data:") :].strip()
                if line == "[DONE]":
                    break

                try:
                    chunk = json.loads(line)
                except Exception:
                    continue

                candidates = chunk.get("candidates") or []
                if not candidates:
                    continue
                cand0 = candidates[0] or {}
                finish_reason = cand0.get("finishReason") or finish_reason
                content = (cand0 or {}).get("content") or {}
                parts = content.get("parts") or []
                for p in parts:
                    t = (p or {}).get("text")
                    if not t:
                        continue
                    if (p or {}).get("thought") is True:
                        thoughts += str(t)
                    else:
                        final += str(t)

                now = time.monotonic()
                if (now - last_ui) >= ui_interval:
                    last_ui = now
                    try:
                        await on_update(thoughts, now - started)
                    except Exception:
                        pass

    try:
        await on_update(thoughts, time.monotonic() - started)
    except Exception:
        pass
    out = (final or "").strip()
    if not out:
        if finish_reason == "MAX_TOKENS":
            raise RuntimeError(
                "Gemini вернул только thinking-часть и упёрся в лимит токенов (MAX_TOKENS), поэтому финального ответа нет.\n"
                "Увеличьте `GEMINI_MAX_OUTPUT_TOKENS` или уменьшите промпт/сложность обработки."
            )
        raise RuntimeError("Gemini не вернул финальный текст (пусто).")
    return out


def _ollama_base_url() -> str:
    return (os.environ.get("OLLAMA_BASE_URL") or DEFAULT_OLLAMA_BASE_URL).strip().rstrip("/")


def _ollama_pick_best_local_model() -> str:
    """
    Best-effort: choose the most capable *locally available* Ollama model.
    Prefers larger models and skips embedding-only / remote models.
    """
    base_url = _ollama_base_url()
    try:
        r = httpx.get(f"{base_url}/api/tags", timeout=10.0)
        r.raise_for_status()
        data = r.json() or {}
    except Exception:
        return ""

    models = data.get("models") or []
    if not isinstance(models, list):
        return ""

    def is_embedding(name: str) -> bool:
        n = (name or "").lower()
        return any(
            t in n
            for t in (
                "embedding",
                "embed",
                "text-embedding",
                "nomic-embed",
                "mxbai-embed",
                "bge-",
            )
        )

    candidates = []
    for m in models:
        if not isinstance(m, dict):
            continue
        # Exclude cloud/remote entries: user asked for local only.
        if (m.get("remote_host") or "").strip():
            continue
        name = (m.get("name") or m.get("model") or "").strip()
        if not name:
            continue
        if is_embedding(name):
            continue
        size = int(m.get("size") or 0)
        candidates.append((size, name))

    if not candidates:
        # Fallback: any local model.
        for m in models:
            if not isinstance(m, dict):
                continue
            if (m.get("remote_host") or "").strip():
                continue
            name = (m.get("name") or m.get("model") or "").strip()
            if not name:
                continue
            size = int(m.get("size") or 0)
            candidates.append((size, name))

    if not candidates:
        return ""
    candidates.sort()
    return candidates[-1][1]


def _ollama_model() -> str:
    raw = (os.environ.get("OLLAMA_MODEL") or "").strip()
    if not raw or raw.lower() == "auto":
        picked = _ollama_pick_best_local_model()
        return picked or DEFAULT_OLLAMA_MODEL
    return raw


def _ollama_options_from_env() -> dict:
    # Keep defaults conservative and suitable for "thinking" models.
    # Users can tune via env vars if needed.
    #
    # Note: On macOS Ollama often uses Metal (GPU) by default, so CPU usage may stay low.
    # These knobs mainly help saturate CPU for CPU-bound configs and/or improve throughput.
    def _f(name: str, default: float) -> float:
        raw = (os.environ.get(name) or "").strip()
        try:
            return float(raw) if raw else default
        except Exception:
            return default

    def _i(name: str, default: int) -> int:
        raw = (os.environ.get(name) or "").strip()
        try:
            return int(raw) if raw else default
        except Exception:
            return default

    def _ram_gb() -> Optional[float]:
        # Best-effort cross-platform physical RAM detection.
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            pages = os.sysconf("SC_PHYS_PAGES")
            if page_size and pages:
                return (float(page_size) * float(pages)) / (1024**3)
        except Exception:
            return None
        return None

    perf = (os.environ.get("OLLAMA_PERF_PROFILE") or "auto").strip().lower()
    cpu_count = int(os.cpu_count() or 0)
    ram_gb = _ram_gb()

    options = {
        "temperature": _f("OLLAMA_TEMPERATURE", 0.7),
        "top_p": _f("OLLAMA_TOP_P", 0.95),
        "num_ctx": _i("OLLAMA_NUM_CTX", 8192),
    }

    # Explicit overrides (highest priority).
    raw_threads = (os.environ.get("OLLAMA_NUM_THREADS") or os.environ.get("OLLAMA_NUM_THREAD") or "").strip()
    raw_batch = (os.environ.get("OLLAMA_NUM_BATCH") or "").strip()
    if raw_threads.isdigit():
        options["num_thread"] = int(raw_threads)
    if raw_batch.isdigit():
        options["num_batch"] = int(raw_batch)

    # Auto-tuning for stronger machines.
    # Only apply when user didn't set explicit overrides.
    if "num_thread" not in options:
        if perf in {"max", "aggressive", "fast"}:
            if cpu_count:
                options["num_thread"] = max(4, cpu_count - 2)
        elif perf == "auto":
            if cpu_count >= 12 and (ram_gb is None or ram_gb >= 32):
                options["num_thread"] = max(4, cpu_count - 2)

    if "num_batch" not in options:
        if perf in {"max", "aggressive", "fast"}:
            # Higher batch can improve throughput on powerful machines.
            options["num_batch"] = 512
        elif perf == "auto":
            if cpu_count >= 12 and (ram_gb is None or ram_gb >= 32):
                options["num_batch"] = 256

    return options


_OLLAMA_PULL_LOCK = threading.Lock()


def _ensure_ollama_model_present(model: str) -> None:
    """
    Best-effort: ensure the model exists locally.
    If OLLAMA_AUTO_PULL=1, runs `ollama pull <model>` when missing.
    """
    base_url = _ollama_base_url()
    try:
        r = httpx.get(f"{base_url}/api/tags", timeout=10.0)
        r.raise_for_status()
        data = r.json() or {}
        models = data.get("models") or []
        if any((m or {}).get("name") == model for m in models):
            return
    except Exception:
        # If Ollama isn't reachable, don't try to pull.
        return

    if not _truthy_env("OLLAMA_AUTO_PULL", True):
        return

    with _OLLAMA_PULL_LOCK:
        # Double-check after waiting for the lock.
        try:
            r = httpx.get(f"{base_url}/api/tags", timeout=10.0)
            r.raise_for_status()
            data = r.json() or {}
            models = data.get("models") or []
            if any((m or {}).get("name") == model for m in models):
                return
        except Exception:
            return
        try:
            subprocess.run(["ollama", "pull", model], check=False)
        except Exception:
            pass


def _split_think_blocks(text: str) -> tuple[str, str]:
    """
    Splits a model output into (thoughts, final) using <think>...</think> blocks.
    Handles the common DeepSeek-R1 style.
    """
    s = text or ""
    thoughts = []
    final_parts = []
    idx = 0
    while True:
        start = s.find("<think>", idx)
        if start == -1:
            final_parts.append(s[idx:])
            break
        final_parts.append(s[idx:start])
        end = s.find("</think>", start + len("<think>"))
        if end == -1:
            thoughts.append(s[start + len("<think>") :])
            break
        thoughts.append(s[start + len("<think>") : end])
        idx = end + len("</think>")
    return ("".join(thoughts).strip(), "".join(final_parts).strip())


async def _ollama_stream_generate(
    *,
    model: str,
    prompt: str,
    system_prompt: Optional[str],
    on_update,
    cancel_event: Optional[asyncio.Event] = None,
) -> str:
    """
    Streams Ollama `/api/chat`. Calls `on_update(thoughts, elapsed_sec)` periodically.
    Returns final answer (tries to strip <think> blocks when present).
    """
    _ensure_ollama_model_present(model)
    url = f"{_ollama_base_url()}/api/chat"
    options = _ollama_options_from_env()
    try:
        logging.getLogger("local_telegram_bot").info(
            "Ollama generate (model=%s perf=%s options=%s)",
            model,
            (os.environ.get("OLLAMA_PERF_PROFILE") or "auto").strip().lower(),
            {k: options.get(k) for k in ("num_thread", "num_batch", "num_ctx", "temperature", "top_p") if k in options},
        )
    except Exception:
        pass
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt or ""},
            {"role": "user", "content": prompt},
        ],
        "stream": True,
        "options": options,
    }

    timeout = httpx.Timeout(connect=5.0, read=600.0, write=30.0, pool=30.0)
    raw = ""
    thoughts = ""
    started = time.monotonic()
    last_ui = 0.0
    ui_interval = 3.0

    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("POST", url, json=payload) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if cancel_event is not None and cancel_event.is_set():
                    raise _UserCancelled()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except Exception:
                    continue
                msg = (chunk or {}).get("message") or {}
                raw += str(msg.get("content") or "")
                th, _final = _split_think_blocks(raw)
                thoughts = th

                now = time.monotonic()
                if (now - last_ui) >= ui_interval:
                    last_ui = now
                    try:
                        await on_update(thoughts, now - started)
                    except Exception:
                        pass

                if (chunk or {}).get("done") is True:
                    break

    try:
        await on_update(thoughts, time.monotonic() - started)
    except Exception:
        pass

    _thoughts, final = _split_think_blocks(raw)
    out = (final or raw).strip()
    if not out:
        raise RuntimeError("Local LLM не вернул финальный текст (пусто).")
    return out


async def handle_process_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.effective_message
    if msg is None or not msg.text:
        return
    text = msg.text.strip()
    low = text.lower()

    # Email auth: if message looks like an email, treat it as an auth request
    # (works without /auth command).
    if _auth_enabled():
        m = re.match(r"^([^@\s]+@[^@\s]+\.[^@\s]+)$", text.strip().lower())
        if m:
            await _begin_email_auth(update, context, email=m.group(1))
            return

    # Email auth: if user has a pending code, try to verify it on any non-command text.
    user = update.effective_user
    if user is not None and _auth_enabled():
        user_id = int(user.id)
        if _auth_has_pending(user_id):
            ok, reply = _auth_verify_code(user_id, text)
            await msg.reply_text(reply)
            return
    return


def _is_group_chat(update: Update) -> bool:
    chat = update.effective_chat
    if not chat or not chat.type:
        return False
    try:
        return chat.type in {ChatType.GROUP, ChatType.SUPERGROUP}
    except Exception:
        # Fallback for older/newer PTB types.
        return str(chat.type).lower() in {"group", "supergroup"}


def _text_mentions_bot(text: str, *, bot_username: Optional[str]) -> bool:
    if not text or not bot_username:
        return False
    return f"@{bot_username.lower()}" in text.lower()


async def handle_group_tag(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Group flow: reply to a voice/audio/video message and mention the bot (e.g. "@botname").
    """
    msg = update.effective_message
    if msg is None or not msg.text:
        return
    if not _is_group_chat(update):
        return

    replied = msg.reply_to_message
    if replied is None:
        return

    bot_username = context.application.bot_data.get("bot_username")
    if not bot_username:
        try:
            me = await context.bot.get_me()
            bot_username = me.username
            if bot_username:
                context.application.bot_data["bot_username"] = bot_username
        except Exception:
            bot_username = None
    if not _text_mentions_bot(msg.text, bot_username=bot_username):
        return

    normalize_audio = _should_normalize_audio(msg.text, bot_username=bot_username)

    try:
        ent_types = ",".join(sorted({(e.type or "") for e in (msg.entities or []) if e})) or "-"
        logging.getLogger("local_telegram_bot").info(
            "Group tag triggered chat=%s msg=%s reply_to=%s entities=%s normalize_audio=%s",
            getattr(update.effective_chat, "id", None),
            getattr(msg, "message_id", None),
            getattr(replied, "message_id", None),
            ent_types,
            "yes" if normalize_audio else "no",
        )
    except Exception:
        pass

    file_id, filename, file_size = _pick_telegram_file_from_message(replied)
    if not file_id:
        await msg.reply_text("Ответьте на voice/audio/video сообщение и упомяните меня (@...).")
        return

    await _process_audio(
        update,
        context,
        file_id=file_id,
        filename=filename or "audio",
        reply_target=msg,
        source_message_id=getattr(replied, "message_id", None),
        source_file_size=file_size,
        normalize_audio=normalize_audio,
    )


async def _process_audio(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    *,
    file_id: str,
    filename: str,
    reply_target,
    source_message_id: Optional[int] = None,
    source_file_size: Optional[int] = None,
    normalize_audio: bool = False,
    local_source_path: Optional[str] = None,
    working_dir: Optional[str] = None,
) -> None:
    """
    Core pipeline used by both private chats (direct voice/audio/video)
    and group chats (reply+mention).
    """
    limit = _asr_concurrency()
    sem: asyncio.Semaphore = context.application.bot_data.setdefault("asr_semaphore", asyncio.Semaphore(limit))
    context.application.bot_data.setdefault("asr_semaphore_limit", limit)

    session_id = uuid.uuid4().hex
    wait_message = None
    wait_stop = asyncio.Event()
    wait_task: Optional[asyncio.Task] = None

    active, pos, sem_limit = await _asr_enqueue(context, session_id=session_id)
    should_show_wait = active >= sem_limit or pos > 1
    if should_show_wait:
        try:
            word = _ru_plural(active, "транскрибация", "транскрибации", "транскрибаций")
            wait_message = await reply_target.reply_text(
                f"⏳ Сейчас обрабатывается {active} {word}.\n"
                f"Ваш файл в очереди под номером {pos}."
            )
            wait_task = asyncio.create_task(
                _queue_wait_updater(context, message=wait_message, session_id=session_id, stop_event=wait_stop)
            )
        except Exception:
            wait_message = None

    try:
        await sem.acquire()
    except asyncio.CancelledError:
        await _asr_dequeue(context, session_id=session_id)
        wait_stop.set()
        if wait_task is not None:
            wait_task.cancel()
        if wait_message is not None:
            await _safe_delete(wait_message)
        raise

    await _asr_dequeue(context, session_id=session_id)
    await _asr_active_inc(context)
    try:
        wait_stop.set()
        if wait_task is not None:
            wait_task.cancel()
        if wait_message is not None:
            await _safe_delete(wait_message)

        cancel_event = asyncio.Event()
        started_at = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).isoformat()
        _active_sessions(context)[session_id] = {
            "cancel_event": cancel_event,
            "user_id": int(getattr(getattr(update, "effective_user", None), "id", 0) or 0),
            "chat_id": int(getattr(getattr(update, "effective_chat", None), "id", 0) or 0),
            "started_at": started_at,
        }
        _write_active_sessions_snapshot(context)
        t0 = time.monotonic()
        user = update.effective_user
        chat = update.effective_chat
        req_msg = update.effective_message
        session: dict = {
            "session_id": session_id,
            "started_at": started_at,
            "trigger": "group_mention" if _is_group_chat(update) else "private_audio",
            "telegram": {
                "user": {
                    "id": getattr(user, "id", None),
                    "username": getattr(user, "username", None),
                    "first_name": getattr(user, "first_name", None),
                    "last_name": getattr(user, "last_name", None),
                    "language_code": getattr(user, "language_code", None),
                },
                "chat": {
                    "id": getattr(chat, "id", None),
                    "type": str(getattr(chat, "type", "") or ""),
                    "title": getattr(chat, "title", None),
                },
                "request_message_id": getattr(reply_target, "message_id", None) or getattr(req_msg, "message_id", None),
                "audio_message_id": source_message_id
                or getattr(reply_target, "message_id", None)
                or getattr(req_msg, "message_id", None),
            },
            "audio": {
                "file_id": file_id,
                "filename": filename,
                "file_size": source_file_size,
                "normalize_audio": bool(normalize_audio),
            },
            "status": "started",
        }
        try:
            whisper_model = _get_env("WHISPER_MODEL", "medium")
            gigaam_model = _get_env("GIGAAM_MODEL", "v3_e2e_rnnt")
            device = _get_env("DEVICE", "cpu")
            language = _get_env("LANGUAGE", "ru")
            hf_token: Optional[str] = (os.environ.get("HF_TOKEN") or "").strip() or None
            gemini_api_key = (os.environ.get("GEMINI_API_KEY") or "").strip()
            gemini_model = (os.environ.get("GEMINI_MODEL") or "gemini-3-pro-preview").strip()
            gemini_system_prompt, prompt_src = _get_system_prompt()
            session["models"] = {
                "whisper": whisper_model,
                "gigaam": gigaam_model,
                "gemini": gemini_model if gemini_api_key else None,
                "device": device,
                "language": language,
                "hf_token": bool(hf_token),
                "llm_system_prompt": prompt_src,
            }

            # Authorization (optional) / whitelist:
            # When AUTH_ENABLED=1, Gemini is allowed only for authorized users,
            # but an authorized user can "whitelist" a group chat on first use.
            user_id0 = int(getattr(user, "id", 0) or 0)
            chat_id0 = int(getattr(chat, "id", 0) or 0)
            auth_enabled = _auth_enabled()
            user_authorized = bool(user_id0 and _auth_is_user_authorized(user_id0))
            chat_whitelisted = bool(chat_id0 and _auth_is_chat_whitelisted(chat_id0))
            whitelisted_now = False
            if auth_enabled and user_id0 and chat_id0 and _is_group_chat(update):
                whitelisted_now = _auth_maybe_whitelist_chat(user_id=user_id0, chat_id=chat_id0)
                if whitelisted_now:
                    chat_whitelisted = True
            can_use_gemini = bool(user_id0 and chat_id0 and ((not auth_enabled) or user_authorized or chat_whitelisted))
            llm_preference = LLM_PROVIDER_GEMINI
            if auth_enabled and user_authorized and user_id0:
                llm_preference = _auth_get_user_llm_provider(user_id0)
            use_gemini = bool(can_use_gemini and gemini_api_key and llm_preference != LLM_PROVIDER_LOCAL)
            session["auth"] = {
                "enabled": auth_enabled,
                "user_authorized": user_authorized,
                "chat_whitelisted": chat_whitelisted,
                "chat_whitelisted_now": whitelisted_now,
                "can_use_gemini": can_use_gemini,
                "llm_preference": llm_preference,
                "use_gemini": use_gemini,
            }

            if user_id0 and not can_use_gemini and _auth_should_prompt_user(user_id0):
                # Don't block: just inform once how to enable Gemini.
                _auth_mark_prompted(user_id0)
                domain = _auth_domain()
                await reply_target.reply_text(
                    f"ℹ️ Gemini доступен только после авторизации по домену {domain}.\n"
                    f"Чтобы авторизоваться, отправьте мне (в личку) вашу почту вида `email@{domain}`."
                )

            def _check_cancel() -> None:
                if cancel_event.is_set():
                    raise _UserCancelled()

            markup = _cancel_markup(session_id)

            # Single "status/progress" message. It is recreated between stages so it's always below previously sent text.
            progress = await reply_target.reply_text("Старт…", reply_markup=markup)
            chat_id = int(update.effective_chat.id) if update.effective_chat else None

            tg_file = None
            source_ready_label = "✅ Аудио подготовлено"
            if local_source_path:
                if not os.path.exists(local_source_path):
                    raise FileNotFoundError(f"Local audio not found: {local_source_path}")
                session["audio"]["local_source"] = os.path.basename(local_source_path)
                session["audio"]["local_source_mode"] = "provided"
            else:
                source_ready_label = "✅ Аудио скачано"
                await _safe_edit_message(progress, "📥 Скачиваю аудио…", reply_markup=markup)
                max_bytes = _telegram_max_get_file_bytes()
                if max_bytes and source_file_size and source_file_size > max_bytes:
                    size_mb = source_file_size / (1024 * 1024)
                    limit_mb = max_bytes / (1024 * 1024)
                    session["status"] = "error"
                    session["error"] = f"incoming_file_too_big({size_mb:.1f}MB>{limit_mb:.0f}MB)"
                    await _safe_delete(progress)
                    await reply_target.reply_text(_file_too_big_message(size_mb=size_mb, limit_mb=limit_mb))
                    return

                try:
                    tg_file = await context.bot.get_file(file_id)
                except BadRequest as exc:
                    if "file is too big" in str(exc).lower():
                        size_mb = (source_file_size or 0) / (1024 * 1024)
                        max_bytes = _telegram_max_get_file_bytes()
                        limit_mb = (max_bytes / (1024 * 1024)) if max_bytes else 20
                        session["status"] = "error"
                        session["error"] = "telegram_get_file_too_big"
                        await _safe_delete(progress)
                        await reply_target.reply_text(_file_too_big_message(size_mb=size_mb, limit_mb=limit_mb))
                        return
                    raise

            tmp_ctx = contextlib.nullcontext(working_dir) if working_dir else tempfile.TemporaryDirectory(prefix="tg_asr_")
            with tmp_ctx as td:
                _check_cancel()
                if not td:
                    raise RuntimeError("No working directory available for transcription.")
                wav_dir = td
                src_path = local_source_path
                if not local_source_path:
                    src_path = os.path.join(td, filename)
                    dl0 = time.monotonic()
                    await tg_file.download_to_drive(custom_path=src_path)
                    _check_cancel()
                    session["timings"] = {"download_sec": round(time.monotonic() - dl0, 3)}
                else:
                    session["timings"] = {"download_sec": 0.0}

                ap = AudioProcessor()
                loop = asyncio.get_running_loop()

                if normalize_audio:
                    await _safe_edit_message(
                        progress,
                        f"{source_ready_label}\n🎚 Нормализую аудио и готовлю WAV (16kHz mono)…",
                        reply_markup=markup,
                    )
                else:
                    await _safe_edit_message(progress, f"{source_ready_label}\n🎛 Готовлю WAV (16kHz mono)…", reply_markup=markup)
                ex0 = time.monotonic()
                audio_filter = _audio_normalize_filter() if normalize_audio else None
                if normalize_audio:
                    try:
                        logging.getLogger("local_telegram_bot").info("Audio normalize enabled (filter=%s)", audio_filter)
                    except Exception:
                        pass
                wav_path = await loop.run_in_executor(
                    None, lambda: ap.extract_audio(src_path, output_dir=wav_dir, audio_filter=audio_filter)
                )
                _check_cancel()
                session["audio"]["wav_path"] = os.path.basename(wav_path)
                session["audio"]["wav_sec"] = round(_wav_duration_sec(wav_path), 3)
                session["timings"]["extract_wav_sec"] = round(time.monotonic() - ex0, 3)

                if normalize_audio:
                    await _safe_edit_message(
                        progress,
                        "✅ WAV готов\n"
                        "🎧 Отправляю обработанное аудио (нормализация + чистка щелчков)…",
                        reply_markup=markup,
                    )
                    _check_cancel()
                    try:
                        processed_path = await loop.run_in_executor(
                            None,
                            lambda: _encode_audio_preview_ffmpeg(
                                ffmpeg_bin=getattr(ap, "ffmpeg_bin", "ffmpeg"),
                                src_wav=wav_path,
                                out_dir=wav_dir,
                                base_name=f"processed_{session_id[:8]}",
                            ),
                        )
                        session["audio"]["processed_audio_path"] = os.path.basename(processed_path)
                        session["audio"]["processed_audio_bytes"] = int(os.path.getsize(processed_path))
                        try:
                            logging.getLogger("local_telegram_bot").info(
                                "Sending processed audio: %s (%d bytes)", processed_path, session["audio"]["processed_audio_bytes"]
                            )
                        except Exception:
                            pass
                        with open(processed_path, "rb") as f:
                            try:
                                await reply_target.reply_audio(
                                    audio=f,
                                    filename=os.path.basename(processed_path),
                                    caption="🎧 Обработанное аудио (нормализация + чистка щелчков)",
                                )
                            except Exception:
                                await reply_target.reply_document(
                                    document=f,
                                    filename=os.path.basename(processed_path),
                                    caption="🎧 Обработанное аудио (нормализация + чистка щелчков)",
                                )
                    except Exception as exc:
                        logging.getLogger("local_telegram_bot").exception("Failed to send processed audio")
                        await reply_target.reply_text(f"⚠️ Не смог отправить обработанное аудио: {str(exc).strip()}")

                    # Recreate progress message so it stays below the sent audio.
                    await _safe_delete(progress)
                    progress = await reply_target.reply_text(
                        f"🧠 GigaAM ({gigaam_model}) — распознаю…\n"
                        "(первый запуск может качать веса)",
                        reply_markup=markup,
                    )
                else:
                    await _safe_edit_message(
                        progress,
                        "✅ WAV готов\n"
                        f"🧠 GigaAM ({gigaam_model}) — распознаю…\n"
                        "(первый запуск может качать веса)",
                        reply_markup=markup,
                    )
                ticker_task = None
                giga_state: Optional[_ProgressState] = None
                if chat_id is not None:
                    audio_sec = _wav_duration_sec(wav_path)
                    giga_state = _ProgressState(audio_sec=audio_sec, est_rtf=_get_rtf_est("gigaam"))
                    ticker_task = asyncio.create_task(
                        _ticker(
                            context=context,
                            chat_id=chat_id,
                            message=progress,
                            base_text=f"🧠 GigaAM ({gigaam_model}) — распознаю…",
                            reply_markup=markup,
                            state=giga_state,
                            cancel_event=cancel_event,
                        )
                    )
                try:
                    giga = GigaAMTranscriptionService(model_name=gigaam_model, device=device, hf_token=hf_token)
                    g0 = time.monotonic()
                    hard_cancel = _truthy_env("GIGAAM_HARD_CANCEL", True)
                    if hard_cancel:
                        try:
                            logging.getLogger("local_telegram_bot").info("GigaAM hard-cancel: enabled (subprocess)")
                        except Exception:
                            pass
                        seg_tuples, _g_info = await _gigaam_transcribe_hard_cancel(
                            wav_path=wav_path,
                            model_name=gigaam_model,
                            device=device,
                            hf_token=hf_token,
                            cancel_event=cancel_event,
                        )
                        g_segments = [ASRSegment(start=a, end=b, text=t) for (a, b, t) in (seg_tuples or [])]
                    else:
                        g_segments, _g_info = await loop.run_in_executor(None, lambda: giga.transcribe(wav_path))
                    g_wall = time.monotonic() - g0
                    if giga_state and giga_state.audio_sec > 0:
                        _update_rtf_est("gigaam", g_wall / giga_state.audio_sec)
                    session["timings"]["gigaam_sec"] = round(g_wall, 3)
                    if isinstance(_g_info, dict):
                        session.setdefault("gigaam", {})["info"] = _g_info
                finally:
                    if ticker_task:
                        ticker_task.cancel()
                        try:
                            await ticker_task
                        except Exception:
                            pass
                _check_cancel()
                g_text = " ".join((s.text or "").strip() for s in g_segments if (s.text or "").strip()).strip()
                session["results"] = {
                    "gigaam_len": len(g_text),
                    "gigaam_segments": len(g_segments or []),
                }

                giga_sent_to_chat = False
                if len(g_text) <= TELEGRAM_TEXT_LIMIT:
                    # Send GigaAM result immediately only when it fits into a regular Telegram message flow.
                    await _safe_delete(progress)
                    progress = None
                    await _reply_long(update, f"GigaAM:\n{g_text or '(пусто)'}")
                    giga_sent_to_chat = True
                    progress = await reply_target.reply_text(
                        f"🧠 Whisper ({whisper_model}) — распознаю…\n"
                        "(первый запуск может качать модель; для скорости beam_size=1)",
                        reply_markup=markup,
                    )
                else:
                    await _safe_edit_message(
                        progress,
                        "ℹ️ Текст GigaAM слишком длинный для сообщения — отправлю его только файлом.\n"
                        f"🧠 Whisper ({whisper_model}) — распознаю…\n"
                        "(первый запуск может качать модель; для скорости beam_size=1)",
                        reply_markup=markup,
                    )

                cpu_threads = int((os.environ.get("WHISPER_CPU_THREADS") or "").strip() or "0")
                whisper = TranscriptionService(
                    model_size=whisper_model,
                    device=device,
                    language=language,
                    cpu_threads=cpu_threads,
                )
                ticker_task = None
                whisper_state: Optional[_ProgressState] = None
                if chat_id is not None:
                    audio_sec = _wav_duration_sec(wav_path)
                    whisper_state = _ProgressState(audio_sec=audio_sec, est_rtf=_get_rtf_est("whisper"))
                    ticker_task = asyncio.create_task(
                        _ticker(
                            context=context,
                            chat_id=chat_id,
                            message=progress,
                            base_text=f"🧠 Whisper ({whisper_model}) — распознаю…",
                            reply_markup=markup,
                            state=whisper_state,
                            cancel_event=cancel_event,
                        )
                    )
                try:
                    audio_sec0 = float(getattr(whisper_state, "audio_sec", 0.0) or 0.0) if whisper_state else 0.0
                    whisper_timeout_env = _parse_timeout_env("WHISPER_TIMEOUT_SEC")
                    if whisper_timeout_env is None:
                        whisper_timeout = _default_whisper_timeout_sec(audio_sec=audio_sec0)
                    else:
                        whisper_timeout = float(whisper_timeout_env)
                    w0 = time.monotonic()
                    cancel_thread_event = threading.Event()

                    async def _mirror_cancel_to_thread() -> None:
                        try:
                            await cancel_event.wait()
                            cancel_thread_event.set()
                        except Exception:
                            pass

                    mirror_task = asyncio.create_task(_mirror_cancel_to_thread())
                    try:
                        if whisper_timeout <= 0:
                            w_segments, _w_info = await loop.run_in_executor(
                                None,
                                lambda: whisper.transcribe(
                                    wav_path,
                                    progress_cb=(whisper_state.set_processed_sec if whisper_state else None),
                                    cancel_cb=cancel_thread_event.is_set,
                                ),
                            )
                        else:
                            w_segments, _w_info = await asyncio.wait_for(
                                loop.run_in_executor(
                                    None,
                                    lambda: whisper.transcribe(
                                        wav_path,
                                        progress_cb=(whisper_state.set_processed_sec if whisper_state else None),
                                        cancel_cb=cancel_thread_event.is_set,
                                    ),
                                ),
                                timeout=float(max(10.0, whisper_timeout)),
                            )
                    except TranscriptionCancelled:
                        raise _UserCancelled()
                    except asyncio.TimeoutError as exc:
                        w_wall = time.monotonic() - w0
                        session["timings"]["whisper_sec"] = round(w_wall, 3)
                        raise RuntimeError(
                            "Whisper не успел завершить распознавание и был остановлен по таймауту.\n"
                            f"Время: {_fmt_dur(w_wall)} / лимит: {_fmt_dur(float(whisper_timeout))}.\n"
                            "Увеличьте `WHISPER_TIMEOUT_SEC` (например, 3600) или отключите таймаут: `WHISPER_TIMEOUT_SEC=0`."
                        ) from exc
                    finally:
                        try:
                            mirror_task.cancel()
                        except Exception:
                            pass
                    w_wall = time.monotonic() - w0
                    if whisper_state and whisper_state.audio_sec > 0:
                        _update_rtf_est("whisper", w_wall / whisper_state.audio_sec)
                    session["timings"]["whisper_sec"] = round(w_wall, 3)
                finally:
                    if ticker_task:
                        ticker_task.cancel()
                        try:
                            await ticker_task
                        except Exception:
                            pass
                _check_cancel()
                w_text = " ".join((s.text or "").strip() for s in w_segments if (s.text or "").strip()).strip()
                session["results"].update(
                    {
                        "whisper_len": len(w_text),
                        "whisper_segments": len(w_segments or []),
                    }
                )
                transcripts_exceed_single_message = (
                    len(w_text) > TELEGRAM_TEXT_LIMIT or len(g_text) > TELEGRAM_TEXT_LIMIT
                )
                session["results"]["transcripts_exceed_single_message"] = transcripts_exceed_single_message
                if transcripts_exceed_single_message:
                    await _safe_edit_message(
                        progress,
                        "ℹ️ Текст распознавания слишком длинный для одного сообщения.\n"
                        "Итог и обе транскрибации пришлю только файлом…",
                        reply_markup=markup,
                    )
                else:
                    await _safe_delete(progress)
                    progress = None
                    if not giga_sent_to_chat:
                        await _reply_long(update, f"GigaAM:\n{g_text or '(пусто)'}")
                    await _reply_long(update, f"Whisper:\n{w_text or '(пусто)'}")

                final_text = ""
                final_error = ""
                final_label = ""

                user_prompt = (
                    "GigaAM:\n"
                    f"{g_text.strip()}\n\n"
                    "Whisper:\n"
                    f"{w_text.strip()}\n"
                )

                if chat_id is not None:
                    await _safe_delete(progress)
                    _check_cancel()

                    llm_sem: asyncio.Semaphore = context.application.bot_data.setdefault(
                        "llm_semaphore", asyncio.Semaphore(_llm_concurrency())
                    )

                    if use_gemini:
                        final_label = f"Gemini ({gemini_model})"
                        progress = await reply_target.reply_text(
                            f"🧠 {final_label} — жду очередь…", reply_markup=markup
                        )
                        gmi0 = time.monotonic()

                        async def on_update(thoughts: str, elapsed_sec: float) -> None:
                            body = _trim_for_telegram(thoughts or "(пока без мыслей)")
                            body_html = _markdown_bold_lines_to_html(body)
                            summary = _active_sessions_summary_line(context)
                            summary_line = f"\n{_html.escape(summary)}" if summary else ""
                            text_html = (
                                f"🧠 { _html.escape(final_label) } — думаю над итогом…\n"
                                f"⏱ {_html.escape(_fmt_dur(elapsed_sec))}{summary_line}\n\n"
                                f"{body_html}"
                            )
                            await _safe_edit_message(progress, text_html, parse_mode=ParseMode.HTML, reply_markup=markup)
                            try:
                                await context.bot.send_chat_action(chat_id=int(chat_id), action=ChatAction.TYPING)
                            except Exception:
                                pass

                        generation_config = {
                            "temperature": _parse_float_env(os.environ.get("GEMINI_TEMPERATURE") or "1", 1.0),
                            "topP": _parse_float_env(os.environ.get("GEMINI_TOP_P") or "0.95", 0.95),
                            "maxOutputTokens": _parse_int_env(
                                os.environ.get("GEMINI_MAX_OUTPUT_TOKENS") or "65536", 65536
                            ),
                            "mediaResolution": _media_resolution_value(os.environ.get("GEMINI_MEDIA_RESOLUTION") or "default"),
                            "thinkingConfig": {
                                "thinkingLevel": _thinking_level_value(os.environ.get("GEMINI_THINKING_LEVEL") or "high"),
                                "includeThoughts": True,
                            },
                        }

                        try:
                            async with llm_sem:
                                await _safe_edit_message(
                                    progress, f"🧠 {final_label} — думаю над итогом…", reply_markup=markup
                                )
                                final_text = await _gemini_stream_generate(
                                    api_key=gemini_api_key,
                                    model=gemini_model,
                                    prompt=user_prompt,
                                    system_prompt=gemini_system_prompt,
                                    generation_config=generation_config,
                                    on_update=on_update,
                                    cancel_event=cancel_event,
                                )
                        except Exception as exc:
                            logging.getLogger("local_telegram_bot").exception("Gemini default processing failed")
                            final_error = str(exc)
                        finally:
                            session["timings"]["llm_sec"] = round(time.monotonic() - gmi0, 3)
                            session["results"]["llm_provider"] = "gemini"
                            session["results"]["llm_model"] = gemini_model
                    else:
                        local_model = _ollama_model()
                        final_label = f"Local ({local_model})"
                        progress = await reply_target.reply_text(
                            f"🧠 {final_label} — жду очередь…", reply_markup=markup
                        )
                        llm0 = time.monotonic()

                        async def on_update(thoughts: str, elapsed_sec: float) -> None:
                            body = _trim_for_telegram(thoughts or "(думаю…)")
                            body_html = _markdown_bold_lines_to_html(body)
                            summary = _active_sessions_summary_line(context)
                            summary_line = f"\n{_html.escape(summary)}" if summary else ""
                            text_html = (
                                f"🧠 { _html.escape(final_label) } — думаю над итогом…\n"
                                f"⏱ {_html.escape(_fmt_dur(elapsed_sec))}{summary_line}\n\n"
                                f"{body_html}"
                            )
                            await _safe_edit_message(progress, text_html, parse_mode=ParseMode.HTML, reply_markup=markup)
                            try:
                                await context.bot.send_chat_action(chat_id=int(chat_id), action=ChatAction.TYPING)
                            except Exception:
                                pass

                        try:
                            async with llm_sem:
                                await _safe_edit_message(
                                    progress, f"🧠 {final_label} — думаю над итогом…", reply_markup=markup
                                )
                                final_text = await _ollama_stream_generate(
                                    model=local_model,
                                    prompt=user_prompt,
                                    system_prompt=gemini_system_prompt,
                                    on_update=on_update,
                                    cancel_event=cancel_event,
                                )
                        except Exception as exc:
                            logging.getLogger("local_telegram_bot").exception("Local LLM processing failed")
                            final_error = str(exc)
                        finally:
                            session["timings"]["llm_sec"] = round(time.monotonic() - llm0, 3)
                            session["results"]["llm_provider"] = "ollama"
                            session["results"]["llm_model"] = local_model
                else:
                    final_error = "Не удалось определить chat_id для LLM."
                    final_label = "LLM"

                if progress:
                    await _safe_edit_message(progress, "📄 Формирую итоговый файл…", reply_markup=markup)

                if final_text and not transcripts_exceed_single_message:
                    chat_text = _gemini_chat_excerpt(final_text)
                    if chat_text.strip():
                        await _reply_long_html(update, _markdown_to_telegram_html(chat_text.strip()))
                elif final_error:
                    await reply_target.reply_text(f"LLM: ошибка/пропуск: {final_error}")

                session["results"].update(
                    {
                        "llm_used": bool(final_text),
                        "llm_error": final_error or None,
                        "llm_len": len(final_text or ""),
                    }
                )

                await _send_markdown_file(
                    update,
                    final_text=final_text,
                    final_error=final_error,
                    final_label=final_label or "LLM",
                    whisper_text=w_text,
                    gigaam_text=g_text,
                )
                session.setdefault("telegram", {}).update(
                    {
                        "sent_transcripts_to_chat": not transcripts_exceed_single_message,
                        "sent_gemini_to_chat": bool(final_text) and not transcripts_exceed_single_message,
                        "sent_markdown_file": True,
                    }
                )

                await _safe_delete(progress)
                await reply_target.reply_text("✅ Готово")
                session["status"] = "ok"
        except _UserCancelled:
            session["status"] = "canceled"
            session["error"] = "canceled_by_user"
            try:
                await _safe_edit_message(progress, "⛔️ Отменено пользователем.", reply_markup=None)
            except Exception:
                pass
        except Exception as exc:
            logging.exception("ASR failed")
            err = str(exc).strip() or exc.__class__.__name__
            await reply_target.reply_text(f"Ошибка: {err}")
            session["status"] = "error"
            session["error"] = err
        finally:
            try:
                _active_sessions(context).pop(session_id, None)
            except Exception:
                pass
            _write_active_sessions_snapshot(context)
            session["ended_at"] = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).isoformat()
            session["timings"] = session.get("timings") or {}
            session["timings"]["total_sec"] = round(time.monotonic() - t0, 3)
            _log_usage_session(session)
    finally:
        try:
            sem.release()
        except Exception:
            pass
        await _asr_active_dec(context)

async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # In groups, process only via reply+mention flow to avoid accidental triggers/spam.
    if _is_group_chat(update):
        return

    file_id, filename, file_size = _pick_telegram_file(update)
    if not file_id:
        await update.effective_message.reply_text("Пришлите voice/audio/video файл.")
        return

    # In private chats, collect media-group items and offer "separate vs merged" for 2+ files.
    if await _handle_media_group_file(
        update,
        context,
        file_id=file_id,
        filename=filename or "audio",
        file_size=file_size,
    ):
        return

    # Use the shared pipeline (also used by reply+mention in group chats).
    await _process_audio(
        update,
        context,
        file_id=file_id,
        filename=filename or "audio",
        reply_target=update.effective_message,
        source_message_id=getattr(update.effective_message, "message_id", None),
        source_file_size=file_size,
    )
    return

    sem: asyncio.Semaphore = context.application.bot_data.setdefault(
        "asr_semaphore", asyncio.Semaphore(_asr_concurrency())
    )
    async with sem:
        try:
            whisper_model = _get_env("WHISPER_MODEL", "medium")
            gigaam_model = _get_env("GIGAAM_MODEL", "v3_e2e_rnnt")
            device = _get_env("DEVICE", "cpu")
            language = _get_env("LANGUAGE", "ru")
            hf_token: Optional[str] = (os.environ.get("HF_TOKEN") or "").strip() or None
            gemini_api_key = (os.environ.get("GEMINI_API_KEY") or "").strip()
            gemini_model = (os.environ.get("GEMINI_MODEL") or "gemini-3-pro-preview").strip()
            gemini_system_prompt, _prompt_src = _get_system_prompt()

            # Single "status/progress" message. It is recreated between stages so it's always below previously sent text.
            progress = await update.effective_message.reply_text("Старт…")
            chat_id = int(update.effective_chat.id) if update.effective_chat else None

            await _safe_edit(progress, "📥 Скачиваю аудио…")
            tg_file = await context.bot.get_file(file_id)

            with tempfile.TemporaryDirectory(prefix="tg_asr_") as td:
                src_path = os.path.join(td, filename)
                wav_dir = td

                await tg_file.download_to_drive(custom_path=src_path)

                ap = AudioProcessor()
                loop = asyncio.get_running_loop()

                await _safe_edit(progress, "✅ Аудио скачано\n🎛 Готовлю WAV (16kHz mono)…")
                wav_path = await loop.run_in_executor(None, lambda: ap.extract_audio(src_path, output_dir=wav_dir))

                await _safe_edit(
                    progress,
                    "✅ WAV готов\n"
                    f"🧠 Whisper ({whisper_model}) — распознаю…\n"
                    "(первый запуск может качать модель; для скорости beam_size=1)",
                )
                cpu_threads = int((os.environ.get("WHISPER_CPU_THREADS") or "").strip() or "0")
                whisper = TranscriptionService(
                    model_size=whisper_model,
                    device=device,
                    language=language,
                    cpu_threads=cpu_threads,
                )
                ticker_task = None
                whisper_state: Optional[_ProgressState] = None
                if chat_id is not None:
                    audio_sec = _wav_duration_sec(wav_path)
                    whisper_state = _ProgressState(audio_sec=audio_sec, est_rtf=_get_rtf_est("whisper"))
                    ticker_task = asyncio.create_task(
                        _ticker(
                            context=context,
                            chat_id=chat_id,
                            message=progress,
                            base_text=f"🧠 Whisper ({whisper_model}) — распознаю…",
                            state=whisper_state,
                        )
                    )
                try:
                    whisper_timeout = int((os.environ.get("WHISPER_TIMEOUT_SEC") or "").strip() or "240")
                    w0 = time.monotonic()
                    w_segments, _w_info = await asyncio.wait_for(
                        loop.run_in_executor(
                            None,
                            lambda: whisper.transcribe(
                                wav_path,
                                progress_cb=(whisper_state.set_processed_sec if whisper_state else None),
                            ),
                        ),
                        timeout=float(max(10, whisper_timeout)),
                    )
                    w_wall = time.monotonic() - w0
                    if whisper_state and whisper_state.audio_sec > 0:
                        _update_rtf_est("whisper", w_wall / whisper_state.audio_sec)
                finally:
                    if ticker_task:
                        ticker_task.cancel()
                        try:
                            await ticker_task
                        except Exception:
                            pass
                w_text = " ".join((s.text or "").strip() for s in w_segments if (s.text or "").strip()).strip()
                await _reply_long(update, f"Whisper:\n{w_text or '(пусто)'}")

                # Recreate status message so it stays below the Whisper output.
                await _safe_delete(progress)
                progress = await update.effective_message.reply_text(
                    f"🧠 GigaAM ({gigaam_model}) — распознаю…\n(первый запуск может качать веса)"
                )

                await _safe_edit(
                    progress,
                    f"🧠 GigaAM ({gigaam_model}) — распознаю…\n(первый запуск может качать веса)",
                )
                giga = GigaAMTranscriptionService(model_name=gigaam_model, device=device, hf_token=hf_token)
                ticker_task = None
                giga_state: Optional[_ProgressState] = None
                if chat_id is not None:
                    audio_sec = _wav_duration_sec(wav_path)
                    giga_state = _ProgressState(audio_sec=audio_sec, est_rtf=_get_rtf_est("gigaam"))
                    ticker_task = asyncio.create_task(
                        _ticker(
                            context=context,
                            chat_id=chat_id,
                            message=progress,
                            base_text=f"🧠 GigaAM ({gigaam_model}) — распознаю…",
                            state=giga_state,
                        )
                    )
                try:
                    g0 = time.monotonic()
                    g_segments, _g_info = await loop.run_in_executor(None, lambda: giga.transcribe(wav_path))
                    g_wall = time.monotonic() - g0
                    if giga_state and giga_state.audio_sec > 0:
                        _update_rtf_est("gigaam", g_wall / giga_state.audio_sec)
                finally:
                    if ticker_task:
                        ticker_task.cancel()
                        try:
                            await ticker_task
                        except Exception:
                            pass
                g_text = " ".join((s.text or "").strip() for s in g_segments if (s.text or "").strip()).strip()
                await _reply_long(update, f"GigaAM:\n{g_text or '(пусто)'}")

                # Gemini post-processing (default pipeline) → markdown file.
                gemini_text = ""
                gemini_error = ""
                if gemini_api_key and chat_id is not None:
                    # Recreate status message so Gemini thoughts are always below the GigaAM output.
                    await _safe_delete(progress)
                    progress = await update.effective_message.reply_text(
                        f"🧠 Gemini ({gemini_model}) — думаю над итогом…"
                    )

                    async def on_update(thoughts: str, elapsed_sec: float) -> None:
                        body = _trim_for_telegram(thoughts or "(пока без мыслей)")
                        body_html = _markdown_bold_lines_to_html(body)
                        text_html = (
                            f"🧠 Gemini ({_html.escape(gemini_model)}) — думаю над итогом…\n"
                            f"⏱ {_html.escape(_fmt_dur(elapsed_sec))}\n\n"
                            f"{body_html}"
                        )
                        await _safe_edit_formatted(progress, text_html, parse_mode=ParseMode.HTML)
                        try:
                            await context.bot.send_chat_action(chat_id=int(chat_id), action=ChatAction.TYPING)
                        except Exception:
                            pass

                    system_prompt = gemini_system_prompt
                    user_prompt = (
                        "Whisper:\n"
                        f"{w_text.strip()}\n\n"
                        "GigaAM:\n"
                        f"{g_text.strip()}\n"
                    )
                    generation_config = {
                        "temperature": _parse_float_env(os.environ.get("GEMINI_TEMPERATURE") or "1", 1.0),
                        "topP": _parse_float_env(os.environ.get("GEMINI_TOP_P") or "0.95", 0.95),
                        "maxOutputTokens": _parse_int_env(os.environ.get("GEMINI_MAX_OUTPUT_TOKENS") or "65536", 65536),
                        "mediaResolution": _media_resolution_value(os.environ.get("GEMINI_MEDIA_RESOLUTION") or "default"),
                        "thinkingConfig": {
                            "thinkingLevel": _thinking_level_value(os.environ.get("GEMINI_THINKING_LEVEL") or "high"),
                            "includeThoughts": True,
                        },
                    }
                    try:
                        gemini_text = await _gemini_stream_generate(
                            api_key=gemini_api_key,
                            model=gemini_model,
                            prompt=user_prompt,
                            system_prompt=system_prompt,
                            generation_config=generation_config,
                            on_update=on_update,
                        )
                    except Exception as exc:
                        logging.getLogger("local_telegram_bot").exception("Gemini default processing failed")
                        gemini_error = str(exc)
                elif not gemini_api_key:
                    gemini_error = "GEMINI_API_KEY не задан — пропускаю обработку Gemini."

                # Send markdown file in requested order.
                if progress:
                    await _safe_edit(progress, "📄 Формирую итоговый файл…")

                # Also send the final Gemini block as a chat message (then the file).
                if gemini_text:
                    chat_text = _gemini_chat_excerpt(gemini_text)
                    if chat_text.strip():
                        await _reply_long_html(update, _markdown_to_telegram_html(chat_text.strip()))
                elif gemini_error:
                    await update.effective_message.reply_text(f"Gemini: ошибка/пропуск: {gemini_error}")

                await _send_markdown_file(
                    update,
                    final_text=gemini_text,
                    final_error=gemini_error,
                    final_label=f"Gemini ({gemini_model})",
                    whisper_text=w_text,
                    gigaam_text=g_text,
                )

                await _safe_delete(progress)
                await update.effective_message.reply_text("✅ Готово")

        except Exception as exc:
            logging.exception("ASR failed")
            await update.effective_message.reply_text(f"Ошибка: {exc}")


def main() -> None:
    load_dotenv()
    # Keep `bot.log` readable: disable third-party progress bars in console.
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TQDM_DISABLE", "1")
    # Reduce noisy third-party warnings in `bot.log`.
    try:
        from urllib3.exceptions import NotOpenSSLWarning

        warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
    except Exception:
        pass
    warnings.filterwarnings("ignore", message=r"Module 'speechbrain\\.pretrained' was deprecated.*")
    setup_logging()
    _setup_usage_logger()
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("snapscript").setLevel(logging.INFO)

    logging.getLogger("local_telegram_bot").info("Python exe=%s prefix=%s", sys.executable, sys.prefix)
    _acquire_bot_lock(os.path.join(os.path.dirname(__file__), ".local_telegram_bot.lock"))

    token = (os.environ.get("TELEGRAM_BOT_TOKEN") or "").strip()
    if not token:
        raise SystemExit("Set TELEGRAM_BOT_TOKEN (or create .env from .env.example).")

    async def post_init(app: Application) -> None:
        try:
            me = await app.bot.get_me()
            if me.username:
                app.bot_data["bot_username"] = me.username
                logging.getLogger("local_telegram_bot").info("Bot username=@%s", me.username)
            else:
                logging.getLogger("local_telegram_bot").warning("Bot has no username; group @mention flow won't work.")
        except Exception:
            logging.getLogger("local_telegram_bot").exception("Failed to fetch bot username via getMe()")

    builder = Application.builder().token(token).post_init(post_init)
    try:
        cu = _bot_concurrent_updates()
        builder = builder.concurrent_updates(cu)
        logging.getLogger("local_telegram_bot").info("Bot concurrent updates: %s", cu)
    except Exception:
        logging.getLogger("local_telegram_bot").exception("Failed to enable concurrent updates; falling back to sequential.")
    app = builder.build()

    async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        err = context.error
        logging.getLogger("local_telegram_bot").exception("Unhandled error: %r", err)

    app.add_error_handler(on_error)
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("auth", cmd_auth))
    app.add_handler(CommandHandler("llm", cmd_llm))
    app.add_handler(CommandHandler("model", cmd_llm))
    app.add_handler(CallbackQueryHandler(cb_media_group_choice, pattern=r"^(mediagroup|twofiles):"))
    app.add_handler(CallbackQueryHandler(cb_cancel, pattern=r"^cancel:"))
    app.add_handler(MessageHandler(filters.ChatType.GROUPS & filters.TEXT & filters.REPLY, handle_group_tag))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_process_text))
    app.add_handler(
        MessageHandler(
            filters.VOICE
            | filters.AUDIO
            | filters.VIDEO
            | filters.VIDEO_NOTE
            | filters.Document.AUDIO
            | filters.Document.VIDEO,
            handle_audio,
        )
    )

    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
