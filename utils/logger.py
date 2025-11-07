# --- Put these imports at top of your module ---
import os, json, logging
from logging.handlers import RotatingFileHandler


# --- One-time logger setup (call once, e.g., in your env __init__) ---
def setup_rl_logger(
    name: str = "rl_transition",
    log_file: str = "rl_transition.jsonl",
    max_bytes: int = 5_000_000,
    backup_count: int = 3,
    level: int = logging.INFO,
):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(level)
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)

    # Rotating JSONL file
    fh = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter("%(message)s"))  # we’ll write JSON strings

    # Console (optional, compact)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger