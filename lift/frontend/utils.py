from __future__ import annotations

import importlib.metadata
import importlib.resources as resources
import json
from typing import Any, Literal

import streamlit as st

from lift.utils import safe_cache_data


def _flatten_dict(d: dict[str, Any], sep: str = ".", _parent_key: str | None = None) -> dict[str, Any]:
    """
    Recursively flatten a nested dictionary into a single-level dictionary with dot-separated keys.

    Instead of chaining dictionary accesses like:
        labels["sidebar"]["general"]["title"]
    you can access the same value as:
        flat_labels["sidebar.general.title"]

    Args:
        d (dict[str, Any]): The nested dictionary to flatten.
        sep (str, optional): Separator between nested keys. Defaults to ".".
        _parent_key (str, optional): Used internally for recursion. Defaults to None.

    Returns:
        dict[str, Any]: Flattened dictionary with dot-separated keys.

    Raises:
        TypeError: If `d` is not a dictionary.

    Examples:
        >>> data = {
        ...     "sidebar": {
        ...         "general": {
        ...             "title": "Allgemeine Parameter",
        ...             "position": {"north": "N", "south": "S"}
        ...         },
        ...         "other": {"enabled": True}
        ...     }
        ... }
        >>> _flatten_dict(data)
        {
            'sidebar.general.title': 'Allgemeine Parameter',
            'sidebar.general.position.north': 'N',
            'sidebar.general.position.south': 'S',
            'sidebar.other.enabled': True
        }
    """
    items = {}
    for k, v in d.items():
        new_key = f"{_parent_key}{sep}{k}" if _parent_key else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, sep=sep, _parent_key=new_key))
        else:
            items[new_key] = v
    return items


def read_json_from_package_data(resource: str, package: str = "lift.data") -> dict:
    """
    Load a JSON file from package resources and return its contents as a dictionary.

    This function reads a JSON file embedded in a Python package using `importlib.resources` and parses it into a
    Python dictionary. It only supports `.json` files.

    Args:
        resource (str): The filename of the JSON resource to load. Must end with ".json".
        package (str, optional): The Python package where the resource is located. Defaults to "lift.data".

    Returns:
        dict: The parsed contents of the JSON file.

    Raises:
        NotImplementedError: If the specified resource filename does not end with ".json".
        FileNotFoundError: If the specified resource cannot be found in the given package.

    Examples:
        >>> data = read_json_from_package_data("labels_en.json")
        >>> type(data)
        <class 'dict'>
        >>> data.get("sidebar")
        {'general': {'title': 'General Parameters', ...}}
    """

    if not resource.endswith(".json"):
        raise NotImplementedError(f'Specified {resource} is not supported. Only ".json" files are supported.')
    try:
        # Use context manager to open the file in text mode
        path_resource = resources.files(package) / resource
        with open(path_resource, "rb") as f:
            return json.load(f)
        return json.loads(resources.read_text(package=package, resource=resource))
    except FileNotFoundError:
        raise FileNotFoundError(f'Specified resource "{resource}" does not exist in package "{package}".')


@safe_cache_data
def _load_language_from_json(language: str) -> dict[str, str]:
    """
    Load and flatten a language JSON file from the package resources.

    This function reads a JSON file corresponding to the specified language from the `lift.data.languages` package,
    flattens it into a single-level dictionary with dot-separated keys, and caches the result for performance.

    Args:
        language (str): The language code to load.
        Supported languages are "de" (German) and "en" (English).

    Returns:
        dict[str, str]: A flattened dictionary of language labels, where nested keys are joined with dots.

    Raises:
        FileNotFoundError: If the specified language is not supported.
    """
    language = language.lower()
    try:
        return _flatten_dict(
            read_json_from_package_data(resource=f"{language}.json", package="lift.data.languages")
            | {"language": language}
        )
    except FileNotFoundError:
        raise FileNotFoundError(f'Specified language "{language}" is not supported.')


def load_language(language: str) -> None:
    """
    Load a language package into the Streamlit session state.

    This function reads the corresponding JSON language file and stores it in ``st.session_state["language_package"]``.
    The selected language code is saved in ``st.session_state["language"]`` for later access by other parts of the app.

    Args:
        language (str): The language code to load.
            Must be either "de" (German) or "en" (English).

    Returns:
        None

    Raises:
        FileNotFoundError: If the specified language JSON file cannot be found.
    """
    st.session_state["language_package"] = _load_language_from_json(language=language)
    st.session_state["language"] = language


def get_label(key: str, language: dict[str, Any] | None = None) -> str:
    """
    Retrieve a label by key from the provided language dict or from session_state.

    Args:
        key (str): Dot-separated label key.
        language (dict[str, Any], optional): A flattened dict containing the labels as values and their keys.
            Defaults to st.session_state["language"] if not provided.

    Returns:
        str: Label value.

    Raises:
        RuntimeError: If no language is provided or loaded into session_state.
        KeyError: If the key is missing in the language package.

    Examples:
        >>> # Use within a streamlit app which `session_state` already contains a language pacakge
        >>> get_label("sidebar.general.title")
        'General Parameters'
        >>> # Explicitly provide language package
        >>> labels = load_language("en")
        >>> get_label("sidebar.general.title", labels)
        'General Parameters'
    """
    if not language:
        language = st.session_state.get("language_package")
        if language is None:
            raise RuntimeError("Language not loaded in session_state. Call load_language() first.")
    if key not in language:
        raise KeyError(f'Label "{key}" not found in language package "{language["language"]}".')
    return language.get(key)


@st.cache_data
def get_version() -> str:
    """
    Retrieve the current version of the `lift` package.

    Returns:
        str: The package version prefixed with 'v', e.g., 'v0.9.0a4'.
             If the package is not installed, returns 'dev'.

    Examples:
        >>> get_version()
        'v0.9.0a4'
    """
    try:
        return f"v{importlib.metadata.version('lift')}"
    except importlib.metadata.PackageNotFoundError:
        return "dev"


def get_supported_languages() -> list[str]:
    """
    Retrieve a list of supported languages based on the JSON files in lift/data/languages.

    Returns:
        list[str]: A list of the supported languages' language codes.

    Examples:
        >>> get_supported_languages()
        ['de', 'en']
    """
    return [f.stem for f in resources.files("lift.data.languages").iterdir() if f.suffix == ".json"]
