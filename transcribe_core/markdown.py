from __future__ import annotations


def render_markdown(
    *,
    final_text: str,
    final_error: str,
    final_label: str,
    whisper_text: str,
    gigaam_text: str,
) -> str:
    if final_error and not final_text:
        final_block = (
            "### Итоговый текст:\n\n"
            f"_({final_label} не смог сформировать итог)_\n\n"
            "### Примечания для контент-менеджера:\n\n"
            "_(нет)_\n\n"
            "### Отчет о проделанных действиях:\n\n"
            f"- Ошибка: `{final_error.strip()}`\n"
        )
    else:
        final_block = (final_text or "").strip()

    return (
        f"## 1) Итоговый вариант по шаблону ({final_label})\n\n"
        f"{final_block}\n\n"
        "---\n\n"
        "## 2) Вариант Whisper\n\n"
        f"{(whisper_text or '').strip()}\n\n"
        "---\n\n"
        "## 3) Вариант GigaAM\n\n"
        f"{(gigaam_text or '').strip()}\n"
    )

