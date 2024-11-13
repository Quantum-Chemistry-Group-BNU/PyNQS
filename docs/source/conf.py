# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'PyNQS'
copyright = '2024, Zhendong Li'
author = 'Zhendong Li'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []
templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

import sphinx_rtd_theme
html_theme = 'sphinx_rtd_theme'
# html_static_path = ['_static']
# html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
extensions = [
    'recommonmark', 
    'sphinx_markdown_tables',
    'sphinxcontrib.tikz',
    'sphinx.ext.mathjax',
]

latex_elements = {
    'preamble': r'''
    \usepackage{tikz},
    \usepackage{physics},
    ''',
}

mathjax3_config = {
    'TeX': {
        'Macros': {
            'braket': [r'\langle #1 | #2 \rangle', 2],
        }
    }
}
