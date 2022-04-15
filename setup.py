#! /usr/bin/env python
"""Computation of persistence Steenrod barcodes"""

import os
import codecs


from pkg_resources.extern.packaging import version
from setuptools import setup, find_packages

version_file = os.path.join('steenroder', '_version.py')
with open(version_file) as f:
    exec(f.read())

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open('docs/requirements.txt') as f:
    doc_requirements = f.read().splitlines()


DISTNAME = 'steenroder'
DESCRIPTION = 'Computation of persistence Steenrod barcodes '
with codecs.open('README.rst', encoding='utf-8-sig') as f:
    LONG_DESCRIPTION = f.read()
LONG_DESCRIPTION_TYPE = 'text/x-rst'
MAINTAINER = 'Anibal Medina Mardones, Umberto Lupo'
MAINTAINER_EMAIL = 'steenroder@gmail.com'
URL = 'https://github.com/Steenroder/steenroder'
LICENSE = 'MIT'
DOWNLOAD_URL = 'https://github.com/Steenroder/steenroder/tarball/v0.1.0'
VERSION = __version__ # noqa
CLASSIFIERS = ['Intended Audience :: Science/Research',
               'Intended Audience :: Developers',
               'License :: OSI Approved',
               'Programming Language :: Python',
               'Topic :: Software Development',
               'Topic :: Scientific/Engineering',
               'Operating System :: Microsoft :: Windows',
               'Operating System :: POSIX',
               'Operating System :: Unix',
               'Operating System :: MacOS',
               'Programming Language :: Python :: 3.8',
               'Programming Language :: Python :: 3.9',
               'Programming Language :: Python :: 3.10']
KEYWORDS = 'topological data analysis, persistent ' + \
    'homology, persistence steenrod modules, persistence ' + \
    'steenrod barcodes'
INSTALL_REQUIRES = requirements
EXTRAS_REQUIRE = {
      'tests': ['pytest',
                'pytest-xdist',
                'nbmake',
                'pytest-cov',
                'flake8'],
      'docs': doc_requirements,
}

setup(name=DISTNAME,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      description=DESCRIPTION,
      license=LICENSE,
      url=URL,
      version=VERSION,
      download_url=DOWNLOAD_URL,
      long_description=LONG_DESCRIPTION,
      long_description_content_type=LONG_DESCRIPTION_TYPE,
      zip_safe=False,
      classifiers=CLASSIFIERS,
      packages=find_packages(),
      keywords=KEYWORDS,
      install_requires=INSTALL_REQUIRES,
      extras_require=EXTRAS_REQUIRE)
