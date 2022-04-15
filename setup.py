#!/usr/bin/python

from setuptools import setup

setup(
    name="fuzzprocess",
    version="0.0.1",
    description="Fuzzy Matching Strings (top k)",
    author="Leo Liu",
    author_email="leo.liu@unsw.edu.au",
    scripts=["fuzzprocess"],
    install_requires=[
        "faiss-gpu",
        "loguru",
        "sentence_transformers",
    ],
)
