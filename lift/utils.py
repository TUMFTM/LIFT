from dataclasses import dataclass
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


@dataclass
class Coordinates:
    latitude: float = 48.148
    longitude: float = 11.507

    @property
    def as_tuple(self) -> tuple[float, float]:
        return self.latitude, self.longitude

    @staticmethod
    def _decimal_to_dms(decimal_deg: float) -> tuple[int, int, float]:
        degrees = int(abs(decimal_deg))
        minutes_full = (abs(decimal_deg) - degrees) * 60
        minutes = int(minutes_full)
        seconds = (minutes_full - minutes) * 60
        return degrees, minutes, seconds

    @property
    def as_dms_str(self) -> str:
        lat_deg, lat_min, lat_sec = self._decimal_to_dms(self.latitude)
        lon_deg, lon_min, lon_sec = self._decimal_to_dms(self.longitude)

        return (
            f"{lat_deg}°{lat_min}'{lat_sec:.2f}'' {'N' if self.latitude >= 0 else 'S'}, "
            f"{lon_deg}°{lon_min}'{lon_sec:.2f}'' {'E' if self.longitude >= 0 else 'W'}"
        )
