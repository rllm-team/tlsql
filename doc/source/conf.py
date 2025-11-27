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

# Add the project root to the path so we can import tl_sql
sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('../../..'))

# -- Project information -----------------------------------------------------

project = 'TL-SQL'
copyright = '2024, TL-SQL Team'
author = 'TL-SQL Team'
version = '0.1.0'
release = '0.1.0'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',        # 自动从代码注释生成文档
    'sphinx.ext.autosummary',     # 自动生成摘要
    'sphinx.ext.viewcode',        # 添加源代码链接
    'sphinx.ext.napoleon',        # 支持 Google/NumPy 风格的 docstring
    'sphinx.ext.intersphinx',    # 链接到其他项目文档
    'sphinx.ext.todo',            # 支持 TODO 注释
    'sphinx.ext.coverage',        # 文档覆盖率
    'sphinx.ext.mathjax',         # 数学公式支持
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
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': False,
    'exclude-members': '__weakref__',
    'show-inheritance': True,
}

# Autosummary settings
autosummary_generate = True
autosummary_imported_members = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Mock imports for modules that may not be available during documentation build
autodoc_mock_imports = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.
html_theme = 'furo'
html_theme_options = {
    "sidebar_hide_name": True,
    "navigation_with_keys": True,
    "light_css_variables": {
        "color-brand-primary": "#3498db",
        "color-brand-content": "#2c3e50",
        "color-background-primary": "#ffffff",
        "color-background-secondary": "#f2f2f2",
    },
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Use default furo sidebar (no custom templates needed)
# html_sidebars = {}  # Use default sidebar configuration

