from setuptools import setup, find_packages
from os import path

from release import read_version

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='cutest',
    version=read_version(string=True),
    description='A cute, composable unit test framework for Python',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/jessebrennan/cutest',
    author='jessebrennan',
    author_email='brennan@pacific.net',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Testing',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    python_requires='>=3.6, <4',
    install_requires=[
        'colorama'
    ],
    entry_points={  # Optional
        'console_scripts': [
            'cutest=cutest:main',
        ],
    },
)
