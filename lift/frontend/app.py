from __future__ import annotations
import os
import traceback

import streamlit as st

# Flag to use streamlit caching; required before importing lift.backend.interfaces
# Is automatically deleted after import
os.environ["LIFT_USE_STREAMLIT_CACHE"] = "1"

from lift.backend.comparison import backend

from lift.backend.simulation.interfaces import GridPowerExceededError, SOCError

# relative imports (e.g. from .design) do not work as app.py is not run as part of the package but as standalone script
from lift.frontend.design import STYLES
from lift.frontend.sidebar import create_sidebar_and_get_input
from lift.frontend.results import display_results, display_empty_results

from lift.frontend.utils import load_language, get_version, get_label

VERSION = get_version()


def display_footer():
    # Inject footer into the page
    # st.markdown(footer, unsafe_allow_html=True)
    sep = '<span style="margin: 0 20px;"> | </span>'
    st.markdown(
        '<div class="footer">'
        "<b>"
        "© 2025 "
        f"<a href='{get_label('footer.institute_url')}' "
        f"target='_blank' "  # open in new tab
        "rel='noopener noreferrer'"  # prevent security and privacy issues with new tab
        f">{get_label('footer.institute')}</a>"
        f" – {get_label('footer.rights')}"
        f"{sep}"
        f"{get_label('footer.version_prefix')}{VERSION}"
        f"{sep}"
        '<a href="https://gitlab.lrz.de/energysystemmodelling/lift" '
        'target="_blank" '  # open in new tab
        'rel="noopener noreferrer"'  # prevent security and privacy issues with new tab
        ">GitLab</a>"
        f"{sep}"
        '<a href="https://www.mos.ed.tum.de/ftm/impressum/" '
        'target="_blank" '  # open in new tab
        'rel="noopener noreferrer"'  # prevent security and privacy issues with new tab
        f">{get_label('footer.imprint')}</a>"
        '<span style="margin: 0 10px;"> </span>'
        "</b></div>",
        unsafe_allow_html=True,
    )


def run_frontend():
    # initialize session state for backend run
    for key in ["run_backend", "auto_refresh"]:
        if key not in st.session_state:
            st.session_state[key] = False

    # initialize language selection
    if "language" not in st.session_state:
        load_language(language="de")

    # define page settings
    st.set_page_config(page_title=get_label("page.title"), page_icon="🚚", layout="wide")

    # css styles for sidebar
    st.markdown(STYLES, unsafe_allow_html=True)

    # create sidebar and get input parameters from sidebar
    settings = create_sidebar_and_get_input()

    st.title(get_label("main.title"))

    if st.session_state["run_backend"] is True:
        try:
            results = backend.run_comparison(comparison_input=settings)
            display_results(results)

        except GridPowerExceededError as e:
            st.error(f"{get_label('main.errors.grid')} {e}")
        except SOCError as e:
            st.error(f"{get_label('main.errors.soc')} {e}")
        except Exception as e:
            st.error(f"{get_label('main.errors.undefined')} {e}")
            st.text(traceback.format_exc())

        st.session_state["run_backend"] = False
    else:
        display_empty_results()

    display_footer()


if __name__ == "__main__":
    run_frontend()
