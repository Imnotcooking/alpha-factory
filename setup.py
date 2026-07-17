from __future__ import annotations

from setuptools import Extension, setup

try:
    import pybind11
except ImportError as exc:  # pragma: no cover - setup-time guard
    raise SystemExit("pybind11 is required to build oqp.native._quant_core") from exc


ext_modules = [
    Extension(
        "oqp.native._quant_core",
        ["src/oqp/native/cpp/quant_core.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=["-O3", "-std=c++17"],
        optional=True,
    )
]


setup(ext_modules=ext_modules)
