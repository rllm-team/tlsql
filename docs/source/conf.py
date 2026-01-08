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

# Configure sys.path for correct tlsql package import
# CRITICAL: The issue is that Python finds tlsql/tlsql/__init__.py instead of tlsql/__init__.py
# Solution: Add the PARENT directory to sys.path, not the tlsql directory itself

import os
import sys

# Get the parent directory that contains the tlsql package
# conf.py is in tlsql/docs/source/, so parent dir contains tlsql/
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))

# FORCE the parent directory to be FIRST in sys.path
# This way, 'import tlsql' finds tlsql/__init__.py in the parent directory
if parent_dir in sys.path:
    sys.path.remove(parent_dir)
sys.path.insert(0, parent_dir)

# Remove the tlsql directory itself if it's in sys.path (prevents confusion)
tlsql_dir = os.path.join(parent_dir, 'tlsql')
while tlsql_dir in sys.path:
    sys.path.remove(tlsql_dir)

# Also remove the subdir to be extra safe
tlsql_subdir = os.path.join(tlsql_dir, 'tlsql')
while tlsql_subdir in sys.path:
    sys.path.remove(tlsql_subdir)

# Ensure tlsql can be imported for viewcode extension
# Force reload to ensure we're using the local version
tlsql_available = False
try:
    # Aggressively clear ALL tlsql-related modules from cache
    modules_to_remove = []
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith('tlsql'):
            modules_to_remove.append(mod_name)

    for mod in modules_to_remove:
        del sys.modules[mod]

    # Also clear any modules that might conflict
    conflict_modules = ['tlsql.tlsql', 'tlsql.core', 'core.tlsql']
    for mod in conflict_modules:
        if mod in sys.modules:
            del sys.modules[mod]

    import tlsql
    import tlsql.tlsql
    import tlsql.tlsql.ast_nodes  # Test import

    # Test that convert function exists
    if not hasattr(tlsql, 'convert'):
        print(f"Warning: tlsql module does not have 'convert' attribute")
        print(f"tlsql module location: {tlsql.__file__}")
        print(f"Available attributes: {[attr for attr in dir(tlsql) if not attr.startswith('_')]}")
    else:
        print(f"✓ Successfully imported tlsql.convert from {tlsql.__file__}")
        tlsql_available = True
except ImportError as e:
    # Print error for debugging but don't fail the build
    print(f"Warning: Could not import tlsql: {e}")
    print("Documentation will be built, but autodoc features may be limited.")
    import traceback
    traceback.print_exc()
except Exception as e:
    # Catch any other errors during import
    print(f"Warning: Unexpected error importing tlsql: {e}")
    print("Documentation will be built, but autodoc features may be limited.")
    import traceback
    traceback.print_exc()

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
napoleon_attr_annotations = False  # Don't show attribute type annotations

# Autodoc settings
autodoc_default_options = {
    'members': True,  # Show members (methods and attributes) by default
    'member-order': 'bysource',
    'special-members': False,  # Don't show special members (__init__, etc.) unless explicitly requested
    'undoc-members': False,  # Don't show members without docstrings
    'exclude-members': '__weakref__',
    'show-inheritance': True,  # Show inheritance in class documentation
    'imported-members': False,  # Don't document imported members to avoid duplicates
    'noindex': False,  # Include in index
}

# Hide type hints in documentation
autodoc_typehints = 'none'  # Don't show type hints in documentation
autodoc_typehints_description_target = 'documented'

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

# Viewcode settings - enable source code links for all documented objects
# This ensures that [source] links appear for all classes and functions
viewcode_enable_epub = True  # Enable viewcode in epub output
viewcode_follow_imported_members = True  # Follow imported members to show their source

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
        "color-brand-primary": "#0194E2",
        "color-brand-secondary": "#19E1C3",
        "color-brand-content": "#2c3e50",
        "color-background-primary": "#ffffff",
        "color-background-secondary": "#f8fbfc",
        "color-background-hover": "#e8f4fd",
        "color-background-border": "#e1e8ed",
        "color-sidebar-background": "#f8fbfc",
        "color-sidebar-background-border": "#e1e8ed",
        "color-sidebar-caption-text": "#6b7280",
        "color-sidebar-link-text": "#374151",
        "color-sidebar-link-text--top-level": "#0194E2",
        "color-sidebar-link-text--hover": "#19E1C3",
        "color-card-background": "#ffffff",
        "color-card-border": "#e1e8ed",
        "color-card-shadow": "rgba(1, 148, 226, 0.1)",
        "color-text-primary": "#111827",
        "color-text-secondary": "#6b7280",
        "color-text-muted": "#9ca3af",
        "font-stack": "system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif",
        "font-stack--monospace": "'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas, 'Courier New', monospace",
    },
    "dark_css_variables": {
        "color-brand-primary": "#4fc3f7",
        "color-brand-secondary": "#81c784",
        "color-brand-content": "#e8f5e8",
        "color-background-primary": "#0f1419",
        "color-background-secondary": "#1a1d23",
        "color-background-hover": "#2a2d32",
        "color-background-border": "#374151",
        "color-sidebar-background": "#161b22",
        "color-sidebar-background-border": "#30363d",
        "color-sidebar-caption-text": "#8b949e",
        "color-sidebar-link-text": "#c9d1d9",
        "color-sidebar-link-text--top-level": "#4fc3f7",
        "color-sidebar-link-text--hover": "#81c784",
        "color-card-background": "#1a1d23",
        "color-card-border": "#30363d",
        "color-card-shadow": "rgba(79, 195, 247, 0.2)",
        "color-text-primary": "#f0f6fc",
        "color-text-secondary": "#c9d1d9",
        "color-text-muted": "#8b949e",
    },
}

# Set the HTML title (shown in browser tab and as home link)
html_title = "TLSQL Documentation"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = ['custom.css']
html_js_files = ['custom.js']

# Use default furo sidebar (no custom templates needed)
# html_sidebars = {}  # Use default sidebar configuration


def setup(app):
    """Setup function for Sphinx."""
    app.add_js_file('custom.js')
