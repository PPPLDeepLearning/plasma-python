import sys
from setuptools import setup, find_packages

#import plasma.version

setup(name = "plasma",
      version = "0.0.1",
      packages = find_packages(),
      #scripts = [""],
      description = "PPPL deep learning package.",
      long_description = """Add description here""",
      author = "Julian Kates-Harbeck, PPPL authors",
      #author_email = "",
      maintainer = "Alexey Svyatkovskiy",
      maintainer_email = "alexeys@princeton.edu",
      #url = "http://",
      download_url = "https://github.com/PPPLDeepLearning/plasma-python",
      #license = "Apache Software License v2",
      test_suite = "tests",
      install_requires = [],
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
