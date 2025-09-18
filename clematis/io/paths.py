import os

def repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def logs_dir() -> str:
    root = repo_root()
    path = os.path.join(root, ".logs")
    os.makedirs(path, exist_ok=True)
    return path
