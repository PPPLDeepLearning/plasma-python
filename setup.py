import os
import sys
import subprocess
from setuptools import setup, find_packages

import plasma.version

try:
    os.environ['MPICC'] = subprocess.check_output("which mpicc", shell=True).decode("utf-8") 
except:
    print ("Please set up the OpenMPI environment")
    exit(1)

setup(name = "plasma",
      version = plasma.version.__version__,
      packages = find_packages(),
      #scripts = [""],
      description = "PPPL deep learning package.",
      long_description = """Add description here""",
      author = "Julian Kates-Harbeck, Alexey Svyatkovskiy",
      author_email = "jkatesharbeck@g.harvard.edu",
      maintainer = "Alexey Svyatkovskiy",
      maintainer_email = "alexeys@princeton.edu",
      #url = "http://",
      download_url = "https://github.com/PPPLDeepLearning/plasma-python",
      #license = "Apache Software License v2",
      test_suite = "tests",
      install_requires = ['keras','pathos','matplotlib','hyperopt','mpi4py'],
      tests_require = [],
      classifiers = ["Development Status :: 3 - Alpha",
                     "Environment :: Console",
                     "Intended Audience :: Science/Research",
                     "Programming Language :: Python",
                     "Topic :: Scientific/Engineering :: Information Analysis",
                     "Topic :: Scientific/Engineering :: Physics",
                     "Topic :: Scientific/Engineering :: Mathematics",
                     "Topic :: System :: Distributed Computing",
                     ],
      platforms = "Any",
      )
