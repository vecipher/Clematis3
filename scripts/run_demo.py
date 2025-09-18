import os
import sys


# Ensure the project root is importable before loading clematis modules.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from clematis.io.config import load_config
from clematis.world.scenario import run_one_turn

def main():
    cfg = load_config(os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml"))
    state = {}
    line = run_one_turn("AgentA", state, "hello world", cfg)
    print("Utterance:", line)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    print("Logs written to:", os.path.join(repo_root, ".logs"))

if __name__ == "__main__":
    # Add repo root to sys.path to simplify running without install
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    main()
