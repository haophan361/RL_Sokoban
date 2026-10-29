from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import sys

if sys.platform == "win32":
    extra_compile_args = [
        "/O2",
        "/std:c++17",
    ]
    extra_link_args = []
else:
    extra_compile_args = [
        "-O3",
        "-std=c++17",
        "-march=native",
        "-ffast-math",
    ]
    extra_link_args = []

sources = [
    "bindings.cpp",
    "src/state.cpp",
    "src/deadlock_detector.cpp",
    "src/sokoban_env.cpp",
    "src/step_info.cpp",
]

ext_modules = [
    Pybind11Extension(
        name="sokoban_cpp",
        sources=sources,
        include_dirs=["include"],
        cxx_std=17,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

setup(
    name="sokoban_cpp",
    version="1.0.0",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
