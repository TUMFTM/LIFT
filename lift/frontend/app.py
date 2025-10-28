from __future__ import annotations
import importlib.metadata
import os
import traceback

import streamlit as st

# Flag to use streamlit caching; required before importing lift.backend.interfaces
# Is automatically deleted after import
os.environ["LIFT_USE_STREAMLIT_CACHE"] = "1"

from lift.backend import backend

from lift.backend.phase_simulation.interfaces import GridPowerExceededError, SOCError

# relative imports (e.g. from .design) do not work as app.py is not run as part of the package but as standalone script
from lift.frontend.design import STYLES
from lift.frontend.sidebar import create_sidebar_and_get_input
from lift.frontend.results import display_results, display_empty_results


@st.cache_data
def get_version() -> str:
    try:
        return f"v{importlib.metadata.version('lift')}"
    except importlib.metadata.PackageNotFoundError:
        return "dev"


VERSION = get_version()


def display_footer():
    # Inject footer into the page
    # st.markdown(footer, unsafe_allow_html=True)
    sep = '<span style="margin: 0 20px;"> | </span>'
    st.markdown(
        '<div class="footer">'
        "<b>"
        "© 2025 "
        '<a href="https://www.mos.ed.tum.de/ftm/" '
        'target="_blank" '  # open in new tab
        'rel="noopener noreferrer"'  # prevent security and privacy issues with new tab
        ">Lehrstuhl für Fahrzeugtechnik, Technische Universität München</a>"
        " – Alle Rechte vorbehalten"
        f"{sep}"
        f"Demo Version {VERSION}"
        f"{sep}"
        '<a href="https://gitlab.lrz.de/energysystemmodelling/lift" '
        'target="_blank" '  # open in new tab
        'rel="noopener noreferrer"'  # prevent security and privacy issues with new tab
        ">GitLab</a>"
        f"{sep}"
        '<a href="https://www.mos.ed.tum.de/ftm/impressum/" '
        'target="_blank" '  # open in new tab
        'rel="noopener noreferrer"'  # prevent security and privacy issues with new tab
        ">Impressum</a>"
        '<span style="margin: 0 10px;"> </span>'
        "</b></div>",
        unsafe_allow_html=True,
    )


def run_frontend():
    # define page settings
    st.set_page_config(
        page_title="LIFT - Logistics Infrastructure & Fleet Transformation", page_icon="🚚", layout="wide"
    )

    # css styles for sidebar
    st.markdown(STYLES, unsafe_allow_html=True)

    # initialize session state for backend run
    for key in ["run_backend", "auto_refresh"]:
        if key not in st.session_state:
            st.session_state[key] = False

    # create sidebar and get input parameters from sidebar
    settings = create_sidebar_and_get_input()

    st.title("LIFT - Logistics Infrastructure & Fleet Transformation")

    if st.session_state["run_backend"] is True:
        try:
            results = backend.run_backend(inputs=settings)
            display_results(results)

        except GridPowerExceededError as e:
            st.error(f"""\
            **Netzanschlussfehler**<br>
            **Der Netzanschluss kann die benötigte Leistung nicht bereitstellen**
            -> Auftretende Lastspitzen können durch einen größeren Netzanschluss oder
            mittels PV-Anlage und stationärem Speicher abgedeckt werden.<br><br>
            Interne Fehlermeldung: {e}
            """)
        except SOCError as e:
            st.error(f"""\
            **Ladezustandsfehler**<br>
            **Der Ladezustand eines Fahrzeugs reicht nicht für die vorgesehene Fahrt aus**<br>
            -> Abhilfe kann eine höhere Ladeleistung (Minimum aus Leistung von Fahrzeug und Ladepunkt), eine höhere
            Anzahl an Ladepunkten oder ein größerer Netzanschluss schaffen.<br><br>
            Interne Fehlermeldung: {e}
            """)
        except Exception as e:
            st.error(f"""\
            **Berechnungsfehler**<br>
            Wenden Sie sich bitte an den Administrator des Tools. Geben Sie dabei die verwendeten Parameter und die
            nachfolgend angezeigte Fehlermeldung an.<br><br>
            Interne Fehlermeldung: {e}
            """)
            st.text(traceback.format_exc())

        st.session_state["run_backend"] = False
    else:
        display_empty_results()

    display_footer()


if __name__ == "__main__":
    run_frontend()
