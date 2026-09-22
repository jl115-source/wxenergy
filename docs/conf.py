from __future__ import annotations

import wxenergy

project = "wxenergy"
author = "Joseph Lockwood"
release = wxenergy.__version__
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
]

autosummary_generate = True
autodoc_typehints = "description"
napoleon_numpy_docstring = True
napoleon_google_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
}

html_theme = "pydata_sphinx_theme"
html_title = "wxenergy"
html_context = {"default_mode": "dark"}
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "navigation_with_keys": True,
    "show_toc_level": 2,
    "navbar_align": "left",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/jl115-source/wxenergy",
            "icon": "fa-brands fa-github",
        }
    ],
}

exclude_patterns = ["_build"]
