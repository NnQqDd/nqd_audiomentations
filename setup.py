from setuptools import setup, find_packages

setup(
    name="nqd-audiomentations",
    version="1.1.4",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "audiomentations",
        "pyrubberband==0.4.0",
        "pyroomacoustics==0.10.0",
        "scipy",
        "librosa"
    ],
)