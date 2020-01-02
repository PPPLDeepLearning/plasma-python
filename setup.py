import os
import subprocess
from setuptools import setup, find_packages

import plasma.version

try:
    os.environ['MPICC'] = subprocess.check_output(
        "which mpicc", shell=True).decode("utf-8")
except BaseException:
    print("Please set up the OpenMPI environment")
    exit(1)

setup(name="plasma",
      version=plasma.version.__version__,
      packages=find_packages(),
      # scripts = [""],
      description="PPPL deep learning package.",
      long_description="""Add description here""",
      author="Julian Kates-Harbeck, Alexey Svyatkovskiy",
      author_email="jkatesharbeck@g.harvard.edu",
      maintainer="Kyle Gerard Felker",
      maintainer_email="felker@anl.gov",
      # url = "http://",
      download_url="https://github.com/PPPLDeepLearning/plasma-python",
      # license = "Apache Software License v2",
      test_suite="tests",
      # TODO(KGF): continue specifying "mininmum reqs" of deps w/o any version
      # info in this file in conjunction with specific reqs in Conda YAML?
      install_requires=[
          'keras',
          'pathos',
          'matplotlib',
          'hyperopt',
          'mpi4py',
          'xgboost',
          'scikit-learn',
          'joblib',
          ],
      # TODO(KGF): add optional feature specs for [deephyper,balsam,
      # readthedocs,onnx,keras2onnx]
      tests_require=[],
      classifiers=["Development Status :: 3 - Alpha",
                   "Environment :: Console",
                   "Intended Audience :: Science/Research",
                   "Programming Language :: Python",
                   "Topic :: Scientific/Engineering :: Information Analysis",
                   "Topic :: Scientific/Engineering :: Physics",
                   "Topic :: Scientific/Engineering :: Mathematics",
                   "Topic :: System :: Distributed Computing",
                   ],
      platforms="Any",
      )
