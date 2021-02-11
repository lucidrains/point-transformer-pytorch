from setuptools import setup, find_packages

setup(
  name = 'point-transformer-pytorch',
  packages = find_packages(),
  version = '0.0.3',
  license='MIT',
  description = 'Point Transformer - Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/point-transformer-pytorch',
  keywords = [
    'artificial intelligence',
    'transformers',
    'attention mechanism',
    'point clouds'
  ],
  install_requires=[
    'einops>=0.3',
    'torch>=1.6'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
