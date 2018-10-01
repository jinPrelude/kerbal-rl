from setuptools import setup

setup(
    name = 'kerbal_rl',
    version = '0.1',
    description = 'Reinforcement learning environment for Kerbal Space Program',
    author = 'Uijin Jung',
    author_email = 'jin.Prelude@gmail.com',
    long_description=open('README.md').read(),
    url = 'https://github.com/jinPrelude/kerbal-rl',
    download_url = 'https://github.com/jinPrelude/kerbal-rl',
    install_requires = [
        'numpy',
        'krpc',
    ],
    python_requires = '>=3'
)