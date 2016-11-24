# -*- coding: utf-8 -*-
"""
    setup.py
    ~~~~~~~~

    Setup script for fileserver

    :copyright: (c) 2015 by Saeed Abdullah.
"""

import os
from setuptools import setup


def read_file(filename):
    """Read file content"""
    with open(filename) as f:
        return f.read()


here = os.path.abspath(os.path.dirname(__file__))
readme = read_file(os.path.join(here, 'README.md'))

setup(
    name='location_analysis',
    version='0.1',
    long_description=readme,
    url='https://github.com/saeed-abdullah/location-analysis',
    author='Saeed Abdullah',
    packages=['location'],
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        'panda==0.18.1',
        'geohash==0.8.5',
        'geopy==1.11.0'
    ],

    dependency_links=['https://github.com/saeed-abdullah/Anvil/tree/master'],

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Topic :: Utilities'
    ]
)
