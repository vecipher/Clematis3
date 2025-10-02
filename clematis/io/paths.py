import os


def repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def logs_dir() -> str:
    env_override = os.getenv("CLEMATIS_LOG_DIR") or os.getenv("CLEMATIS_LOGS_DIR")
    if env_override:
        path = os.path.abspath(os.path.expanduser(env_override))
    else:
        root = repo_root()
        path = os.path.join(root, ".logs")
    os.makedirs(path, exist_ok=True)
    return path
