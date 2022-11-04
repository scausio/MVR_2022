from setuptools import setup, find_packages


setup(name='pqtool',
      version='0.1',
      description='Product quality toolbox for Black Sea validation',
      packages=find_packages(),
      install_requires=['numpy', 'scipy', 'netCDF4', 'xarray', 'intake', 'gsw', 'luigi'])
