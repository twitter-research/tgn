from setuptools import setup, find_packages

setup(
    name='TGN',
    version='0.0.1',
    description='Temporal Graph Nets. Graph Neural Networks for dynamic data.',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'torch',
        'scikit_learn'
    ],
)
