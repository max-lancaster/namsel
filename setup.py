from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

import platform

if platform.system() == "Windows":
    setup(
      name = 'Namsel OCR',
      version = '16.10',
      description = 'A system for performing OCR on machine-printed Tibetan text',
      url = 'http://www.namsel.com',
      author = 'Zach Rowinski',
      author_email = 'zach.rowinski@gmail.com',
      
      license = 'MIT',
      
      install_requires=['PIL', 'numpy', 'sklearn', 'cython', 'cairo', 'pango', 'pangocairo'],
      
      cmdclass = {'build_ext': build_ext},
      ext_modules = [Extension("sobel_features", [r"features\sobel_features.pyx"],
                              include_dirs=[np.get_include()],
                              extra_compile_args=['-O2'],
                              ),
                     Extension("zernike_moments", [r"features\zernikeh.pyx"],
                              include_dirs=[np.get_include()],
                              extra_compile_args=['-O2'],
                              ),
                     Extension("transitions", [r"features\transitions.pyx"],
                              include_dirs=[np.get_include()],
                              extra_compile_args=['-O2'],
                              ),
                     Extension("viterbi_cython", ["viterbi_cython.pyx"],
                              include_dirs=[np.get_include()],
                              extra_compile_args=['-O2'],
                              ),
                     Extension("fast_utils", ["fast_utils.pyx"],
                              include_dirs=[np.get_include()],
                              extra_compile_args=['-O2'],
                              ),
#                     Extension("features", ["features.pyx"],
#                              include_dirs=[np.get_include()],
#                              extra_compile_args=['-O3', '-ffast-math'],
#                              )
                              ])
else:
    setup(
      name = 'Namsel OCR',
      version = '16.10',
      description = 'A system for performing OCR on machine-printed Tibetan text',
      url = 'http://www.namsel.com',
      author = 'Zach Rowinski',
      author_email = 'zach.rowinski@gmail.com',
      
      license = 'MIT',
      
      install_requires=['PIL', 'numpy', 'sklearn', 'cython', 'cairo', 'pango', 'pangocairo'],
      
      cmdclass = {'build_ext': build_ext},
      ext_modules = [Extension("sobel_features", ["features/sobel_features.pyx"],
                              include_dirs=[np.get_include()],
                              extra_compile_args=['-O2'],
                              ),
                     Extension("zernike_moments", ["features/zernikeh.pyx"],
                              include_dirs=[np.get_include()],
                              extra_compile_args=['-O2'],
                              ),
                     Extension("transitions", ["features/transitions.pyx"],
                              include_dirs=[np.get_include()],
                              extra_compile_args=['-O2'],
                              ),
                     Extension("viterbi_cython", ["viterbi_cython.pyx"],
                              include_dirs=[np.get_include()],
                              extra_compile_args=['-O2'],
                              ),
                     Extension("fast_utils", ["fast_utils.pyx"],
                              include_dirs=[np.get_include()],
                              extra_compile_args=['-O2'],
                              ),
#                     Extension("features", ["features.pyx"],
#                              include_dirs=[np.get_include()],
#                              extra_compile_args=['-O3', '-ffast-math'],
#                              )
                              ])
