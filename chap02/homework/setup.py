from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

# compile_flags = ['-std=c++11', '-fopenmp']
# linker_flags = ['-fopenmp']

# module = Extension('tm',
#                    ['tm.pyx'],
#                    language='c++',
#                    include_dirs=[numpy.get_include()],  # This helps to create numpy
#                    extra_compile_args=compile_flags,
#                    extra_link_args=linker_flags)
# setup(
#     name='tm',
#     ext_modules=cythonize(module),
#     zip_safe=False,
#     gdb_debug=True  # This is extremely dangerous; Set it to False in production.
# )

# Mac不支持openmp
compile_flags = ['-std=c++11']
module = Extension('tm',
                   ['tm.pyx'],
                   language='c++',
                   include_dirs=[numpy.get_include()],  # This helps to create numpy
                   extra_compile_args=compile_flags)
setup(
    name='tm',
    ext_modules=cythonize(module)
)