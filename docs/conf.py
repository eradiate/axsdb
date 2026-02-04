# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import axsdb

project = "axsdb"
copyright = "2025, Vincent Leroy"
author = "Vincent Leroy"
version = axsdb.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # Core extensions
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    # Third-party
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_iconify",
    "autodocsumm",
]

templates_path = ["_templates"]
source_suffix = [".rst", ".md"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "cachetools": ("https://cachetools.readthedocs.io/en/stable/", None),
    "pint": ("https://pint.readthedocs.io/en/stable/", None),
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


html_static_path = ["_static"]
html_title = "axsdb"

# Use Shibuya theme
# https://shibuya.lepture.com/
html_theme = "shibuya"
html_theme_options = {
    "accent_color": "orange",
    "navigation_with_keys": True,
    "github_url": "https://github.com/eradiate/axsdb",
}
html_logo = "_static/logo_axsdb.svg"
