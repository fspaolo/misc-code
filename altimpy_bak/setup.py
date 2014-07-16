import sys, os
from setuptools import setup, find_packages

NAME = 'altim'
VERSION = '0.1'

HERE = os.path.abspath(os.path.dirname(__file__))
README = open(os.path.join(HERE, 'README.rst')).read()
NEWS = open(os.path.join(HERE, 'NEWS.txt')).read()

install_requires = [
    # List your project dependencies here.
    # For more details, see:
    # http://packages.python.org/distribute/setuptools.html#declaring-dependencies
]

setup(name=NAME,
    version=VERSION,
    description='Set of tools for processing satelllite altimetry data',
    long_description=README + '\n\n' + NEWS,
    classifiers=[
      # Get strings from http://pypi.python.org/pypi?%3Aaction=list_classifiers
    ],
    keywords='satellite data, altimetry, time series, IDR, WDR',
    author='Fernando Paolo',
    author_email='fspaolo@gmail.com',
    url='',
    license='MIT',
    packages=find_packages('altim', 'altim.io'),
)

