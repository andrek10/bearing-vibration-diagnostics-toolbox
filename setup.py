from setuptools import setup

setup(
    name='pyvib',
    version='1.0.0',
    author='Andreas Klausen',
    author_email='andreas.klausen@motiontech.no',
    entry_points={
    },
    scripts=[
    ],
    url='https://github.com/andrek10/bearing-vibration-diagnostics-toolbox',
    license='LICENSE.md',
    description='PyVib Python Library',
    install_requires=[
        'numpy>=1.19.2',
        'pyfftw>=0.12.0',
        'numba>=0.54.1',
        'scipy>=1.7.3',
        'scikit-learn>=1.0.1',
        'pandas>=1.3.4',
        'matplotlib>=3.5.0',
        'psutil>=5.8.0',
        'netcdf4>=1.5.8'
    ],
)
