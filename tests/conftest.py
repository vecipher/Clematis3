# Ensure the repo root (which contains the 'clematis' package) is on sys.path for tests.
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)