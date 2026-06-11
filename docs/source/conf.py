# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'PyNQS'
copyright = '2024-2026, Zhendong Li'
author = 'Zhendong Li'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []
templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# import sphinx_rtd_theme
html_theme = 'sphinx_rtd_theme'
# html_static_path = ['_static']
# html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
extensions = [
    "myst_parser",
    'sphinx_markdown_tables',
    'sphinxcontrib.tikz',
    'sphinx.ext.mathjax',
]

latex_elements = {
    'preamble': r'''
    \usepackage{tikz}
    ''',
}

mathjax3_config = {
    'TeX': {
        'Macros': {
            'braket': [r'\langle #1 | #2 \rangle', 2],
            'ket':  [r'|#1\rangle', 1],
            'kett': [r'|#1\rangle\!\rangle', 1],
            'bra':  [r'\langle#1|', 1],
            'braa': [r'\langle\!\langle#1|', 1],
            'ev':   [r'\langle #1\rangle', 1],
            'mel':  [r'\langle #1|#2|#3\rangle', 3],
            'pdv':  [r'\frac{\partial #1}{\partial #2}', 2],
            'ee':   r'\mathrm{e}',
            'Cov':  r'\operatorname{Cov}',
            'ii':   r'\mathrm{i}',
            'Pf':   r'\operatorname{Pf}',
        }
    }
}
