from setuptools import setup, find_namespace_packages

setup(name='phtrans',
      packages=find_namespace_packages(include=["phtrans", "phtrans.*"]),
      version='0.0.1',
      install_requires=[
           "wandb"
      ],
      entry_points={
          'console_scripts': [
              'PHTrans_train = phtrans.run.run_training:main',
          ],
      },
      )
