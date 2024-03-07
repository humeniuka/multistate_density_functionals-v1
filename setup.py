#!/usr/bin/env python
from os import path
import re
from io import open
from setuptools import setup

def get_property(property, package):
    result = re.search(
        r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format(property),
        open(path.join('src', package, '__init__.py')).read(),
    )
    return result.group(1)

this_dir = path.abspath(path.dirname(__file__))
with open(path.join(this_dir, 'README.rst'), encoding='utf8') as f:
    long_description = f.read()

setup(
    name='msdft',
    version=get_property('__version__', 'msdft'),
    description='Functionals for multistate density functional theory',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    url='https://github.com/humeniuka/multistate_density_functionals',
    author='Alexander Humeniuk',
    author_email='alexander.humeniuk@gmail.com',
    license='LICENSE.txt',
    package_dir = {"": "src"},
    install_requires=['matplotlib', 'numpy', 'pyscf==2.4.0', 'scipy', 'tqdm'],
    include_package_data=True,
    zip_safe=False,
)
