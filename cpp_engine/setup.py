from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        'quant_core', # The name of the module we will import in Python
        ['quant_core.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=['-std=c++11', '-O3'] # -O3 turns on maximum C++ speed optimization
    ),
]

setup(
    name='quant_core',
    version='1.0',
    description='C++ Accelerated Quant Engine',
    ext_modules=ext_modules,
)