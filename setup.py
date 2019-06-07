#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = [  # 'opencv-python-headless',  # for SIFT functionality MUST conda install opencv or build source
    'aicsimageio',
    'cython>=0.29.5',
    'imageio',
    'numpy>=1.16.1',
    'openpiv',
    'pandas',
    'scikit-video',
    'matplotlib<3.0.0'
    # 'opencv-python',
]

test_requirements = [
    'codecov',
    'flake8',
    'pytest',
    'pytest-cov',
    'pytest-raises',
]

setup_requirements = ['pytest-runner', ]

dev_requirements = [
    'bumpversion>=0.5.3',
    'wheel>=0.33.1',
    'flake8>=3.7.7',
    'tox>=3.5.2',
    'coverage>=5.0a4',
    'Sphinx>=2.0.0b1',
    'twine>=1.13.0',
    'pytest>=4.3.0',
    'pytest-cov==2.6.1',
    'pytest-raises>=0.10',
    'pytest-runner>=4.4',
]

interactive_requirements = [
    'altair',
    'jupyterlab',
    'matplotlib',
]

extra_requirements = {
    'test': test_requirements,
    'setup': setup_requirements,
    'dev': dev_requirements,
    'interactive': interactive_requirements,
    'all': [
        *requirements,
        *test_requirements,
        *setup_requirements,
        *dev_requirements,
        *interactive_requirements
    ]
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
    entry_points={
        'console_scripts': [
            'generate_deformation_map=aicsdeformation.bin.deformation_map:main',
            'optimize_paramenters=aicsdeformation.bin.optimize_deformation:main'
        ],
    },
    install_requires=requirements,
    license="Allen Institute Software License",
    long_description=readme,
    include_package_data=True,
    keywords='aicsdeformation',
    name='aicsdeformation',
    packages=find_packages(exclude=['tests', '*.tests', '*.tests.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    extras_require=extra_requirements,
    url='https://github.com/AllenCellModeling/aicsdeformation',
    version='0.1.0',
    zip_safe=False,
)
