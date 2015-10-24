#!/usr/bin/python
import sys
from setuptools import setup

sys.path.append('./tpmrec')
sys.path.append('./tests')

setup(
    name='tpmrec',
    version='1.0.0',
    description='A flexible framework of topic model and recommender system',
    packages=['tpmrec'],
    install_requires=['numpy'],
    test_suite='tests'
)
