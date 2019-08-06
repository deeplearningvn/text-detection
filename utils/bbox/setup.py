from distutils.core import setup, Extension

import numpy as np
from Cython.Build import cythonize

numpy_include = np.get_include()

# setup(ext_modules=[
#     Extension('bbox', ['bbox.pyx'], include_dirs=[numpy_include]),
#     Extension('nms', ['nms.pyx'], include_dirs=[numpy_include])
# ])

# extensions = [Extension("*", ["*.pyx"])]

setup(ext_modules=cythonize([
      Extension('bbox', ['bbox.pyx'], include_dirs=[numpy_include]),
      Extension('nms', ['nms.pyx'], include_dirs=[numpy_include])
      ]))


# setup(ext_modules=cythonize("bbox.pyx"), include_dirs=[numpy_include])
# setup(ext_modules=cythonize("nms.pyx"), include_dirs=[numpy_include])
