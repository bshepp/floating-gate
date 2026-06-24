"""Put the repo root on sys.path so flat modules (verdict.py, ...) import directly in tests."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
