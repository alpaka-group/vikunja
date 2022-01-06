# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os, subprocess, shutil

# -- Project information -----------------------------------------------------

project = 'vikunja'
copyright = '2021, Simeon Ehrig'
author = 'Simeon Ehrig'

# The full version, including alpha/beta/rc tags
release = '[0.2.0]'


# -- General configuration ---------------------------------------------------

# build on readthedocs.io
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autosectionlabel'
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
html_theme = 'sphinx_rtd_theme'

html_logo = "logo/vikunja_logo.svg"
html_theme_options = {
    "logo_only"  : True
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
#html_static_path = ['_static']

# -- Doxygen build -----------------------------------------------------------

if on_rtd:
    rel_sphinx_build_folder_path = '_build'
else:
    rel_sphinx_build_folder_path = '../build'

# path where doxygen generate html files
doxygen_build_dir = '../build/doxygen/html'
# build folder of sphinx -> conntet is online available
doxygen_dst = os.path.join(rel_sphinx_build_folder_path, 'html/doxygen')

# create build folder, if not already exists
if not os.path.exists(doxygen_build_dir):
    os.makedirs(doxygen_build_dir)

# create doxygen documentation
print("copy doxygen from {} to {}".format(doxygen_build_dir, doxygen_dst))
subprocess.call('cd ..; doxygen Doxyfile', shell=True)

if os.path.exists(doxygen_dst):
    shutil.rmtree(doxygen_dst)
shutil.copytree(src=doxygen_build_dir, dst=doxygen_dst)