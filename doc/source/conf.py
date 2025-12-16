# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

# Add the project root to the path so we can import tlsql
tlsql_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if tlsql_dir not in sys.path:
    sys.path.insert(0, tlsql_dir)
parent_dir = os.path.abspath(os.path.join(tlsql_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# -- Project information -----------------------------------------------------

project = 'TLSQL'
copyright = '2024, TLSQL Team'
author = 'TLSQL Team'
version = '0.1.0'
release = '0.1.0'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',        # Automatically generate docs from code comments
    'sphinx.ext.autosummary',    # Automatically generate summaries
    'sphinx.ext.viewcode',       # Add source code links
    'sphinx.ext.napoleon',       # Support Google/NumPy style docstrings
    'sphinx.ext.intersphinx',    # Link to other project docs
    'sphinx.ext.todo',           # Support TODO comments
    'sphinx.ext.coverage',       # Documentation coverage
    'sphinx.ext.mathjax',        # Math formula support
]

# Napoleon settings for docstring parsing
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Autodoc settings
autodoc_default_options = {
    'members': False,  # Don't show members in sidebar, only show class names
    'member-order': 'bysource',
    'special-members': False,  # Don't show special members
    'undoc-members': False,
    'exclude-members': '__weakref__',
    'show-inheritance': False,  # Don't show inheritance in sidebar
    'imported-members': False,  # Don't document imported members to avoid duplicates
}

# Autosummary settings
autosummary_generate = False  # Disable autosummary to reduce clutter in sidebar
autosummary_imported_members = False  # Don't document imported members to avoid duplicates

# Hide module names in class/function signatures
add_module_names = False  # Don't show module names in class/function titles

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'conversion.rst']

# Mock imports for modules that may not be available during documentation build
autodoc_mock_imports = []

# Intersphinx mapping for external documentation
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'sqlalchemy': ('https://docs.sqlalchemy.org/en/20/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.
html_theme = 'furo'
html_theme_options = {
    "sidebar_hide_name": False,  # Show project name in sidebar (acts as home link)
    "navigation_with_keys": True,
    "light_css_variables": {
        "color-brand-primary": "#3498db",
        "color-brand-content": "#2c3e50",
        "color-background-primary": "#ffffff",
        "color-background-secondary": "#f2f2f2",
    },
}

# Set the HTML title (shown in browser tab and as home link)
html_title = "TLSQL Documentation"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Use default furo sidebar (no custom templates needed)
# html_sidebars = {}  # Use default sidebar configuration
