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
# conf.py is in doc/source/, so go up two levels to reach tlsql/ directory
# Then go up one more level to reach the parent directory that contains tlsql/
tlsql_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
parent_dir = os.path.abspath(os.path.join(tlsql_dir, '..'))

# Only add parent_dir to sys.path (not tlsql_dir)
# This ensures that 'import tlsql' imports tlsql/__init__.py (which has convert)
# and 'import tlsql.tlsql' imports tlsql/tlsql/__init__.py
# If we add tlsql_dir to sys.path, 'import tlsql' would import tlsql/tlsql/__init__.py instead
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Ensure tlsql can be imported for viewcode extension
# Force reload to ensure we're using the local version
try:
    # Remove tlsql from sys.modules if it exists to force reload
    modules_to_remove = [key for key in list(sys.modules.keys()) if key.startswith('tlsql')]
    for module_name in modules_to_remove:
        del sys.modules[module_name]
    
    # Ensure we use the source code version by checking the path
    # The parent_dir should contain tlsql/ directory
    tlsql_source_path = os.path.join(parent_dir, 'tlsql', '__init__.py')
    if os.path.exists(tlsql_source_path):
        print(f"Using source code from: {tlsql_source_path}")
        # The parent_dir is already in sys.path, so 'import tlsql' should work
        # But we need to make sure it's at the front of the path
        if parent_dir in sys.path:
            sys.path.remove(parent_dir)
        sys.path.insert(0, parent_dir)
    
    # Try to import tlsql
    # If it fails due to FilterCondition import error from installed package,
    # try to uninstall and use source code instead
    try:
        import tlsql
        print(f"Imported tlsql from: {tlsql.__file__}")
        
        # Check if we're using installed package with old FilterCondition import
        actual_path = os.path.abspath(tlsql.__file__)
        expected_path = os.path.abspath(os.path.join(parent_dir, 'tlsql', '__init__.py'))
        
        if actual_path != expected_path:
            # We're using installed package, check if it has the old import
            try:
                with open(actual_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                if 'FilterCondition' in content:
                    print(f"Warning: Installed package at {actual_path} contains FilterCondition import")
                    print("Attempting to use source code instead...")
                    # Try to uninstall the package
                    import subprocess
                    try:
                        subprocess.check_call(
                            [sys.executable, '-m', 'pip', 'uninstall', '-y', 'tlsql'],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL
                        )
                        print("Uninstalled old tlsql package")
                    except:
                        pass
                    
                    # Remove from sys.modules and try again
                    modules_to_remove = [key for key in list(sys.modules.keys()) if key.startswith('tlsql')]
                    for module_name in modules_to_remove:
                        del sys.modules[module_name]
                    
                    # Ensure source code path is first
                    if parent_dir in sys.path:
                        sys.path.remove(parent_dir)
                    sys.path.insert(0, parent_dir)
                    
                    import tlsql
                    print(f"Now using source code from: {tlsql.__file__}")
            except:
                pass  # If we can't read the file, continue anyway
    except ImportError as import_err:
        error_msg = str(import_err)
        if 'FilterCondition' in error_msg:
            print(f"Detected FilterCondition import error: {error_msg}")
            print("Attempting to uninstall old package and use source code...")
            
            # Try to uninstall the package
            import subprocess
            try:
                subprocess.check_call(
                    [sys.executable, '-m', 'pip', 'uninstall', '-y', 'tlsql'],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                print("Uninstalled old tlsql package")
            except:
                pass
            
            # Remove from sys.modules
            modules_to_remove = [key for key in list(sys.modules.keys()) if key.startswith('tlsql')]
            for module_name in modules_to_remove:
                del sys.modules[module_name]
            
            # Ensure source code path is first
            if parent_dir in sys.path:
                sys.path.remove(parent_dir)
            sys.path.insert(0, parent_dir)
            
            # Try importing again
            try:
                import tlsql
                print(f"Successfully imported tlsql from source: {tlsql.__file__}")
            except ImportError:
                print(f"Still cannot import tlsql after uninstall attempt")
                raise
    
    # Try to import submodules, but don't fail if they don't exist
    try:
        import tlsql.tlsql
    except ImportError:
        print("Warning: Could not import tlsql.tlsql submodule")
    
    try:
        import tlsql.tlsql.ast_nodes  # Test import
    except ImportError:
        print("Warning: Could not import tlsql.tlsql.ast_nodes")
    
    # Test that convert function exists
    if not hasattr(tlsql, 'convert'):
        print(f"Warning: tlsql module does not have 'convert' attribute")
        print(f"tlsql module location: {tlsql.__file__}")
        print(f"tlsql module attributes: {dir(tlsql)}")
    else:
        print(f"✓ Successfully imported tlsql.convert from {tlsql.__file__}")
except ImportError as e:
    # Print error for debugging but don't fail
    print(f"Warning: Could not import tlsql: {e}")
    import traceback
    traceback.print_exc()
    pass
except Exception as e:
    # Handle any other errors gracefully
    print(f"Warning: Error during tlsql import setup: {e}")
    import traceback
    traceback.print_exc()
    pass

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
        "color-brand-primary": "#0066cc",
        "color-brand-content": "#1a1a1a",
        "color-background-primary": "#ffffff",
        "color-background-secondary": "#f8f9fa",
        "color-sidebar-background": "#f8f9fa",
        "color-sidebar-background-border": "#e0e0e0",
        "color-sidebar-caption-text": "#666666",
        "color-sidebar-link-text": "#1a1a1a",
        "color-sidebar-link-text--top-level": "#0066cc",
        "font-stack": "system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif",
        "font-stack--monospace": "'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas, 'Courier New', monospace",
    },
    "dark_css_variables": {
        "color-brand-primary": "#4da6ff",
        "color-brand-content": "#e0e0e0",
        "color-background-primary": "#1a1a1a",
        "color-background-secondary": "#242424",
        "color-sidebar-background": "#1e1e1e",
        "color-sidebar-background-border": "#333333",
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
