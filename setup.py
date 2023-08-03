import io
import os
import re
import subprocess
from typing import List, Set

import setuptools
import pybind11
from pybind11.setup_helpers import Pybind11Extension


ROOT_DIR = os.path.dirname(__file__)

CXX_FLAGS = ["-g", "-O2", "-std=c++17", f"-I{pybind11.get_include()}"]
#NVCC_FLAGS = ["-O2", "-std=c++17"]


ext_modules = []

test_1_module = Pybind11Extension(
    "honx._min_dist",
    ["src/honx/_min_dist/dijkstra.cpp",],
    #include_dirs=[pybind_include_dir],
)
ext_modules.append(test_1_module)


def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)


def find_version(filepath: str):
    """Extract version information from the given filepath.

    Adapted from https://github.com/ray-project/ray/blob/0b190ee1160eeca9796bc091e07eaebf4c85b511/python/setup.py
    """
    with open(filepath) as fp:
        version_match = re.search(
            r"^__version__ = ['\"]([^'\"]*)['\"]", fp.read(), re.M)
        if version_match:
            return version_match.group(1)
        raise RuntimeError("Unable to find version string.")


def read_readme() -> str:
    """Read the README file."""
    return io.open(get_path("README.md"), "r", encoding="utf-8").read()


def get_requirements() -> List[str]:
    """Get Python package dependencies from requirements.txt."""
    with open(get_path("requirements.txt")) as f:
        requirements = f.read().strip().split("\n")
    return requirements


setuptools.setup(
    name="honx",
    version=find_version(get_path("src/honx", "__init__.py")),
    author="honglu",
    license="Apache 2.0",
    description="jit-compilable honk honk",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/honglu2875/honx",
    project_urls={
        "Homepage": "https://github.com/honglu2875/honx",
        #"Documentation": "",
    },
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=get_requirements(),
    ext_modules=ext_modules,
)