from setuptools import setup, find_packages

setup(
    name='expy',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'scipy>=1.7.0',
        'matplotlib>=3.4.0',
        
    ],
   
    author='Domingos de Eulária Dumba',
    description = 'A module to facilitate the design and analysis of statistical experiments',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
)