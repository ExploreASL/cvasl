# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, under development

import sys, subprocess
import os


sys.path.insert(0, os.path.abspath('..'))
# -- Project information -----------------------------------------------------

project = 'cvasl'
copyright = '2023, c.moore@esciencecenter.nl'
author = 'c.moore@esciencecenter.nl'

# The full version, including alpha/beta/rc tags
try:
    tag = subprocess.check_output([
        'git',
        '--no-pager',
        'describe',
        '--abbrev=0',
        '--tags',
    ]).strip().decode()
except subprocess.CalledProcessError as e:
    print(e.output)
    tag = 'v0.1.1'

release = tag[1:]


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.extlinks',
    'sphinx.ext.imgmath',
    'sphinx.ext.intersphinx',

]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'piccolo_theme'

html_theme_options = {
    #"bgcolor": "grey", # Background color.
    # "headfont (CSS font-family): Font for headings.
}

# Add any paths that contain custom static files (such as style sheets) here, #
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static/basic_mod.css']


intersphinx_mapping = {
    'python': (
        'https://docs.python.org/{.major}'.format(
            sys.version_info,
        ),
        None,
    ),
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    'matplotlib': ('http://matplotlib.org', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'IPython': ('https://ipython.readthedocs.io/en/stable/', None),
}

