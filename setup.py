from setuptools import setup

version = "0.0.1"

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
)
