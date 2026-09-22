from __future__ import annotations

import wxenergy

project = "wxenergy"
author = "Joseph Lockwood"
release = wxenergy.__version__
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
]

autosummary_generate = True
autodoc_typehints = "description"
napoleon_numpy_docstring = True
napoleon_google_docstring = False

html_theme = "pydata_sphinx_theme"
html_title = "wxenergy"
html_context = {"default_mode": "dark"}
html_theme_options = {
    "navigation_with_keys": True,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/jl115-source/wxenergy",
            "icon": "fa-brands fa-github",
        }
    ],
}

exclude_patterns = ["_build"]
