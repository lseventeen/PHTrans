from setuptools import setup, find_namespace_packages

setup(name='phtrans',
      packages=find_namespace_packages(include=["phtrans", "phtrans.*"]),
      version='0.0.1',
      install_requires=[
           "wandb",
           "thop",
           "timm",
           "einops"
      ],
      entry_points={
          'console_scripts': [
              'PHTrans_train = phtrans.run.run_training:main',
              'PHTrans_BCV = phtrans.dataset_conversion.Task017_BeyondCranialVaultAbdominalOrganSegmentation:main',
              'PHTrans_ACDC = phtrans.dataset_conversion.Task027_AutomaticCardiacDetectionChallenge:main',
              'PHTrans_predict = phtrans.inference.predict_simple:main',
              
          ],
      },
      )
