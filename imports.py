"""
Centralised imports used across the analysis scripts.

Keep this file lightweight and dependency tolerant.
Optional libraries are imported with try/except so the pipeline can run even if
a non essential package is missing.
"""

from __future__ import annotations

from pathlib import Path
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from wordcloud import WordCloud
from matplotlib.ticker import FixedLocator, FixedFormatter

# Optional libraries (present in the original notebook but not required for the plots below)
try:
    import folium  # noqa: F401
except Exception:
    folium = None

try:
    from geopy.geocoders import Nominatim  # noqa: F401
    from geopy.extra.rate_limiter import RateLimiter  # noqa: F401
except Exception:
    Nominatim = None
    RateLimiter = None


TOKEN_PATTERN = r"(?u)\b[a-zA-Z]{3,}\b"

CUSTOM_STOP = {"rt", "amp", "via", "http", "https", "tco", "co"}


def get_stop_words() -> list[str]:
    """Standard English stop words plus a small custom list."""
    return list(set(ENGLISH_STOP_WORDS) | CUSTOM_STOP)


def safe_name(s: str, max_len: int = 120) -> str:
    """Convert free text to a filesystem safe name."""
    s = "" if s is None else str(s).strip()
    if not s:
        return "unknown"
    return re.sub(r"[^\w]+", "_", s)[:max_len]


def project_paths(repo_root: Path | None = None) -> dict[str, Path]:
    """
    Resolve canonical repository paths.

    repo_root defaults to the directory containing this file.
    """
    root = repo_root or Path(__file__).resolve().parent
    data_dir = root / "data"
    outputs_dir = root / "outputs"
    return {"root": root, "data": data_dir, "outputs": outputs_dir}
