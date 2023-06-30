#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = [
    "transformers>=4.30.2",
    "accelerate>=0.20.3",
    "datasets>=2.12.0",
    "peft>=0.3.0",
    "bitsandbytes>=0.39.1",
    "pyopenssl>=23.2.0",
    "tqdm",
    "codecarbon",
]

test_requirements = []

setup(
    author="Giuseppe Attanasio",
    author_email="giuseppeattanasio6@gmail.com",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="Simple generation interface for prompting HF CausalLM and Seq2Seq models.",
    install_requires=requirements,
    license="Apache Software License 2.0",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="simple_generation",
    name="simple_generation",
    packages=find_packages(include=["simple_generation", "simple_generation.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/g8a9/simple_generation",
    version="0.1.0",
    zip_safe=False,
)
