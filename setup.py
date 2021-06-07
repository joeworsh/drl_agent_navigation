# Copyright Joe Worsham 2021

from setuptools import setup, find_packages

setup(name='joe-agents',
      version='0.1.0',
      description='DRL Agents for Udacity Navigation Project',
      license='MIT',
      author='Joe Worsham',
      url='https://github.com/joeworsh/drl_agent_navigation',
      packages=find_packages(),
      install_requires=[
            "gym", "numpy", "torch", "tqdm"
      ]
)
