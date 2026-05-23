import logging
import os
import sys
from logging.handlers import RotatingFileHandler

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "app.log")


def setup_logger():
    root = logging.getLogger()
    if root.handlers:
        return
    root.setLevel(logging.DEBUG)
    # Silence verbose network and model download logging from Hugging Face / HTTPX.
    # This keeps terminal output focused on app-level events only.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

    fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    if hasattr(ch.stream, "reconfigure"):
        try:
            ch.stream.reconfigure(encoding="utf-8", errors="backslashreplace")
        except Exception:
            pass

    fh = RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    root.addHandler(ch)
    root.addHandler(fh)


def get_logger(name: str):
    setup_logger()
    return logging.getLogger(name)
