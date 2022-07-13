# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"

from os.path import realpath, dirname, join
import types
import sys

# Include 'src/' in PYTHONPATH
# in order to import the 'Ammolite' package
doc_path = dirname(realpath(__file__))
package_path = join(dirname(doc_path), "src")
sys.path.insert(0, package_path)
import springcraft

# Include springcraft/doc in PYTHONPATH
# in order to import modules for example generation etc.
sys.path.insert(0, doc_path)
import scraper


#### General ####

extensions = ["sphinx.ext.autodoc",
              "sphinx.ext.autosummary",
              "sphinx.ext.doctest",
              "sphinx.ext.mathjax",
              "sphinx.ext.viewcode",
              "sphinx_gallery.gen_gallery",
              "numpydoc"]

templates_path = ["templates"]
source_suffix = [".rst"]
master_doc = "index"

project = "springcraft"
copyright = "2022, the Springcraft contributors."
version = springcraft.__version__

exclude_patterns = ["build"]

numpydoc_show_class_members = False
autodoc_default_options = {
    "show-inheritance": True
}

pygments_style = "sphinx"

todo_include_todos = False

# Prevents numpydoc from creating an autosummary which does not work
# properly due to Biotite's import system
numpydoc_show_class_members = False

autodoc_member_order = "bysource"


#### HTML ####

html_theme = "alabaster"
html_static_path = ["static"]
html_css_files = [
    "biotite.css",
    "fonts.css",
]
html_favicon = "static/assets/springcraft_logo_32p.png"
htmlhelp_basename = "SpringcraftDoc"
html_theme_options = {
    "description"   : "Investigate molecular dynamics with elastic network models",
    "logo"          : "assets/springcraft_logo.svg",
    "logo_name"     : "true",
    "github_user"   : "biotite-dev",
    "github_repo"   : "springcraft",
    "github_banner" : "true",
    "github_type"   : "star",
    "fixed_sidebar" : "true",
    "page_width"    : "1200px",
}
sphinx_gallery_conf = {
    "examples_dirs"             : "examples/scripts",
    "gallery_dirs"              : "examples/gallery",
    'filename_pattern'          : "",
    "download_all_examples"     : False,
    # Never report run time
    "min_reported_time"         : sys.maxsize,
    "image_scrapers"            : ("matplotlib", scraper.pymol_scraper,),
    # Replace 'ammolite.show()'
    "reset_modules"             : (scraper.overwrite_display_func,),
    # Do not capture file path string output
    # by the overwritten 'ammolite.show()'
    "capture_repr"              : (),
}


#### App setup ####

def skip_non_methods(app, what, name, obj, skip, options):
    if skip:
        return True
    if what == "class":
        # Functions
        if type(obj) in [
            types.FunctionType, types.BuiltinFunctionType, types.MethodType
        ]:
            return False
        return True


def setup(app):
    app.connect("autodoc-skip-member", skip_non_methods)