from pathlib import Path

import pytest

pytest.importorskip("sphinx")

from sphinx.application import Sphinx

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"


def test_docs_build_has_no_warnings(tmp_path):
    """The Sphinx build must succeed with warnings promoted to errors.

    Regression test for docstrings whose numpydoc ``Parameters`` sections
    (bare ``**kwargs`` entries, parameters with no description line) parse as
    broken raw RST instead of being rendered by Napoleon, and for Sphinx
    config entries (e.g. ``html_static_path``) pointing at missing paths --
    both silently pass a plain ``sphinx-build`` but fail CI's ``-W`` build.
    """
    app = Sphinx(
        srcdir=str(DOCS_DIR),
        confdir=str(DOCS_DIR),
        outdir=str(tmp_path / "html"),
        doctreedir=str(tmp_path / "doctrees"),
        buildername="html",
        warningiserror=True,
        keep_going=True,
    )
    app.build()
    assert app.statuscode == 0
