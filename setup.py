import os
import sys
import re

try:
    from setuptools import setup

    setup
except ImportError:
    from distutils.core import setup

    setup

version = "0.1.0"

setup(
    name="gasp",
    version=version,
    packages=["gasp"],
    setup_requires=[
        "pytest-runner",
    ],
    tests_require=[
        "pytest",
    ],
    package_data={"gasp": ["data/filters/*par"]},
    include_package_data=True,
)
