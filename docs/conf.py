"""Sphinx configuration for omnicast."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "Omnicast"
copyright = "2026, Omnicast contributors"
author = "Omnicast contributors"

from omnicast import __version__ as release

version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx_autodoc_typehints",
    "myst_nb",
]

myst_enable_extensions = ["colon_fence", "deflist"]

# The real-data notebook already carries its own saved outputs (plots included) from a
# run against the live CMU Delphi Epidata API; re-executing it on every docs build would
# make the build depend on network access and the optional `epidatpy` package for no
# benefit, so myst-nb only renders what's already there.
nb_execution_mode = "off"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "show-inheritance": True,
    "undoc-members": False,
}
autosummary_generate = True
napoleon_google_docstring = True
napoleon_numpy_docstring = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
}

html_theme = "furo"
html_static_path = ["_static"]
html_title = "Omnicast"
