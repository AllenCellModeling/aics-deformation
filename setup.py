#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from pip._internal import main as pipmain
from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'cython>=0.29.5',
    'numpy>=1.16.1',
    'openpiv>=0.21.2',
    'tifffile>=2019.2.10',
]

setup_requirements = [
    'pytest-runner>=4.4',
]

test_requirements = [
    'pytest>=4.3.0',
    'pytest-cov==2.6.1',
    'pytest-raises>=0.10',
]

extra_requirements = {
    'test': test_requirements,
    'setup': setup_requirements,
}

setup(
    author="Jackson Maxfield Brown",
    author_email='jacksonb@alleninstitute.org',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: Allen Institute Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="Processing and visualization tools used for interacting with deformation projects.",
    entry_points={},
    install_requires=requirements,
    license="Allen Institute Software License",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='aicsdeformation',
    name='aicsdeformation',
    packages=find_packages(include=['aicsdeformation']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    extras_require=extra_requirements,
    url='https://github.com/AllenCellModeling/aicsdeformation',
    version='0.1.0',
    zip_safe=False,
)

#######################################################################################################################

# Handle full installation by overriding `setup` with `_pre_install`
pre_install_requirements = [
    'cython>=0.29.5',
    'numpy>=1.16.1',
]


# Preinstall any dependancies
def _pre_install(setup):
    def install(package):
        pipmain(['install', package])

    for package in pre_install_requirements:
        install(package)

    return setup


# Override setup with _pre_install
setup = _pre_install(setup)
