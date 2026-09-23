"""Pytest path bootstrap: make the mini_DBMS root importable regardless of
how pytest is invoked."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
