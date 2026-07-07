from setuptools import setup, find_packages

setup(
    name="nqd-audiomentations",
    version="1.1.7",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "audiomentations",
        "pyroomacoustics==0.10.0",
        "scipy",
        "librosa"
    ],
)