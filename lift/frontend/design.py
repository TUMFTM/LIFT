"""Frontend design helpers (colors, CSS) for Streamlit app.

Purpose:
- Load custom color palette from `.streamlit/colors.toml` and provide CSS snippets
  for consistent theming (sidebar width, footer, headings).

Relationships:
- Colors consumed by `frontend/results.py` and other UI modules; styles injected in `frontend/app.py`.
- Uses Streamlit caching to avoid repeated file reads.
"""

import importlib.resources
from typing import Tuple

import toml

from lift.utils import safe_cache_data


# Load specified project colors from colors.toml
@safe_cache_data
def get_colors() -> Tuple[str, str, str, str]:
    # Get custom colors from config.toml
    with importlib.resources.files("lift").joinpath(".streamlit/colors.toml").open("r") as f:
        config = toml.load(f)
    colors = config.get("custom_colors", {})
    return colors["tumblue"], colors["baseline"], colors["expansion"], colors["lightblue"]


COLOR_TUMBLUE, COLOR_BL, COLOR_EX, COLOR_LIGHTBLUE = get_colors()

STYLES = f"""
    <style>
        /* Hide the sidebar collapse button */
        div[data-testid="stSidebarCollapseButton"] {{
            display: none !important;
        }}

        /* Style for fixed footer */
        .footer {{
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            width: 100%;
            background-color: {COLOR_TUMBLUE};
            color: white;
            text-align: right;
            padding: 10px;
            font-size: 14px;
        }}

        /* Remove link styling inside the footer */
        .footer a {{
            color: inherit;
            text-decoration: none;
        }}

        .footer a:hover {{
            text-decoration: underline;
        }}

        /* Define style for sidebar */
        [data-testid="stSidebar"] {{
            min-width: 450px;
            max-width: 450px;
            width: 450px;
        }}
        [data-testid="stSidebarContent"] {{
            padding-right: 20px;
        }}
        div[data-testid="stSidebar"] button {{
            width: 100% !important;
        }}

        /* remove anchor link icons from headlines */
        /* 1) Markdown-headlines (#, ##, ### ...) */
        [data-testid="stMarkdownContainer"] h1 > a,
        [data-testid="stMarkdownContainer"] h2 > a,
        [data-testid="stMarkdownContainer"] h3 > a,
        [data-testid="stMarkdownContainer"] h4 > a,
        [data-testid="stMarkdownContainer"] h5 > a,
        [data-testid="stMarkdownContainer"] h6 > a,
        [data-testid="stMarkdownContainer"] h1 svg,
        [data-testid="stMarkdownContainer"] h2 svg,
        [data-testid="stMarkdownContainer"] h3 svg,
        [data-testid="stMarkdownContainer"] h4 svg,
        [data-testid="stMarkdownContainer"] h5 svg,
        [data-testid="stMarkdownContainer"] h6 svg {{
            display: none !important;
            visibility: hidden !important;
            pointer-events: none !important;
        }}

        /* 2) component headlines (st.header/subheader/title) */
        [data-testid="stHeading"] a,
        [data-testid="stHeading"] svg,
        [data-testid="stHeadingWithAnchor"] a,
        [data-testid="stHeadingWithAnchor"] svg {{
            display: none !important;
            visibility: hidden !important;
            pointer-events: none !important;
        }}

        /* Remove top padding/margin of main page and set top area to be transparent*/
        header.stAppHeader {{
            background-color: transparent;
        }}
        section.stMain .block-container {{
            padding-top: 0rem;
            z-index: 1;
        }}

    </style>
    """

LINE_HORIZONTAL = "<hr style='margin-top: 0.1rem; margin-bottom: 0.5rem;'>"
