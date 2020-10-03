from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy
from sphinx.setup_command import BuildDoc

version = "0.0.1"

setup(name="gasp", version=version, packages=["gasp"])
