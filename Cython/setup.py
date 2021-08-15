from distutils.core import Extension, setup
from Cython.Build import cythonize


ext = Extension(name="module", sources=["module.pyx"])
setup(ext_modules=cythonize(ext, language_level = "3"))