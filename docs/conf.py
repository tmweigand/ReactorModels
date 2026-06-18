# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys

# Make the installed package importable by autodoc
sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------------
project = "ReactorModels"
copyright = "2026, Timothy M. Weigand"
author = "Timothy M. Weigand"
release = "0.1"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "myst_parser",
    "sphinx.ext.imgmath",  # ← added: renders math via real LaTeX
]

autodoc_member_order = "bysource"
napoleon_google_docstring = True
napoleon_numpy_docstring = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- LaTeX / imgmath configuration
_sty_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "latex"))

imgmath_latex_preamble = r"""
\usepackage{amsmath}
\makeatletter
\let\proof\relax
\let\endproof\relax
\makeatother
""" + f"\\makeatletter\\input{{{_sty_dir}/ctmmath-v3.sty}}\\makeatother"


# Optional tweaks
imgmath_image_format = "svg"  # "svg" (sharp) or "png" (wider support)
imgmath_font_size = 13  # match your HTML body font size

# Keep the latex_* keys for PDF builds via `make latexpdf`
latex_additional_files = ["latex/ctmmath-v3.sty"]
latex_elements = {
    "preamble": r"""
\usepackage{ctmmath-v3}
""",
}

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
