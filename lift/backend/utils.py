"""Backend utility functions for caching and environment detection.

Purpose:
- Provides `safe_cache_data`, a decorator that conditionally uses Streamlit caching when running in a
  Streamlit context, and acts as a no-op otherwise (e.g., in batch scripts or tests).

Relationships:
- Used by `backend/comparison/comparison.py`, `backend/evaluation/evaluation.py`, and
  `backend/simulation/simulation.py` to memoize deterministic function results.
- The caching flag is set by `frontend/app.py` via environment variable before importing backend modules.

Key Logic:
- Detects Streamlit context by checking the `LIFT_USE_STREAMLIT_CACHE` environment variable (set by frontend).
- If Streamlit is available, wraps functions with `st.cache_data` for automatic memoization.
- If not (e.g., in CLI/batch mode), returns the function unchanged, ensuring backend code works in both contexts.
"""

from functools import wraps
import os

import streamlit as st


# store flag in current process and delete environment variable
# flag is only set by frontend.py and therefore only exists when a streamlit app is running
_use_streamlit_cache = os.environ.pop("LIFT_USE_STREAMLIT_CACHE", "0") == "1"


def safe_cache_data(*dargs, **dkwargs):
    """
    Safe replacement for st.cache_data:
    - In Streamlit app: uses st.cache_data
    - Outside Streamlit: no-op
    Supports both @safe_cache_data and @safe_cache_data(...)
    """

    def decorator(func):
        if _use_streamlit_cache:
            return st.cache_data(**dkwargs)(func)
        else:

            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapper

    # Used without arguments -> @safe_cache_data
    if dargs and callable(dargs[0]):
        return decorator(dargs[0])

    # Used with arguments -> @safe_cache_data(...)
    return decorator
