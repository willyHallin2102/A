"""
    Loader
    debugging loader
"""
from __future__ import annotations

import os, sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.append(str(root))

import argparse
