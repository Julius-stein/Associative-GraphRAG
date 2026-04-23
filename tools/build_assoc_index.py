#!/usr/bin/env python3
"""Build an Association-compatible corpus index without importing `lightrag`."""

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from associative_rag_project.index_builder import main


if __name__ == "__main__":
    main()
