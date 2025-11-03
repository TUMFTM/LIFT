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
