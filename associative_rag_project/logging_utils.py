"""Minimal logging helpers for long-running retrieval / generation jobs.

提供简单日志输出与文本截断工具，便于观测长时运行过程。
"""

from datetime import datetime


def log(message):
    """Print a timestamped log line that is easy to scan in terminal runs."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def shorten(text, limit=120):
    """Collapse whitespace and trim long previews for logs and debug output."""
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."
