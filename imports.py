"""
Imports all libraries required for the project
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

TOKEN_PATTERN = r"(?u)\b[a-zA-Z]{3,}\b"
CUSTOM_STOP = {"rt", "amp", "via", "http", "https", "tco", "co"}


def get_stop_words() -> list[str]:
    """Standard English stop words plus a small custom list."""
    return list(set(ENGLISH_STOP_WORDS) | CUSTOM_STOP)


def safe_name(s: str, max_len: int = 120) -> str:
    s = "" if s is None else str(s).strip()
    if not s:
        return "unknown"
    return re.sub(r"[^\w]+", "_", s)[:max_len]
