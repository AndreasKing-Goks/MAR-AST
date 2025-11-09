# --- Put these imports at top of your module ---
import os, json, logging, datetime
from logging.handlers import RotatingFileHandler
import numpy as np

def _to_plain(obj):
    """Make args values JSON/print friendly."""
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (set,)):
        return sorted(list(obj))
    return obj

def log_ast_training_config(
    args,
    txt_path: str,
    env=None,
    also_print: bool = False,
    section_title: str = "AST TRAINING CONFIG"
):
    """
    Append AST training configuration (from argparse.Namespace `args`)
    into the same log file as env.log_RL_transition_text.

    Parameters
    ----------
    args : argparse.Namespace
        Your parsed CLI args.
    txt_path : str
        The same base path you pass to env.log_RL_transition_text (without .txt).
    env : gym.Env | None
        Optional; if provided, we’ll log action/observation space info.
    also_print : bool
        Echo the same lines to console.
    section_title : str
        Header title for this section.
    """
    # Build lines
    lines = []
    lines.append(f'#============================================ {section_title} ===========================================#')
    lines.append(f'timestamp                     : {datetime.datetime.now().isoformat(timespec="seconds")}')
    
    # Dump args in a stable, readable order
    if args is not None:
        # Convert Namespace to plain dict then sanitize values
        if hasattr(args, "__dict__"):
            args_dict = vars(args)
        else:
            # If already a dict-like
            args_dict = dict(args)
        # sanitize and sort keys for readability
        cleaned = {k: _to_plain(v) for k, v in args_dict.items()}
        for k in sorted(cleaned.keys()):
            v = cleaned[k]
            # Pretty one-line JSON when it’s a container; plain otherwise
            if isinstance(v, (list, dict)):
                v_str = json.dumps(v, ensure_ascii=False)
            else:
                v_str = str(v)
            lines.append(f'{k:30s}: {v_str}')
    else:
        lines.append('args                          : None')

    # Optional env metadata (nice to have)
    if env is not None:
        lines.append('#----------------------------------- Used AST Environment Wrapper -------------------------------------#')
        lines.append(f'env id/name                   : {getattr(env, "spec", None) and getattr(env.spec, "id", None) or type(env).__name__}')

    # Write to file (append)
    log_file = (txt_path or "") + ".txt"
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    if also_print:
        for line in lines:
            print(line)

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