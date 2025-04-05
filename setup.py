# setup.py required for building your ml application
# as a package itself
from setuptools import find_packages, setup
from typing import List


HYPEN_E_DOT = '-e .'
def get_requirements(file_path):
    requirements=[]
    with open("requirements.txt", "r") as file:
        requirements=file.readlines()
        requirements = [pkg.replace("\n","") for pkg in requirements]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements

setup(
name='mlproject',
version='0.1',
author='Ritesh Jain',
author_email='jainritesh2876@email.com',
description='A simple example of ml pipeline',
packages=find_packages(),
install_requires=get_requirements("requirements.txt")
)