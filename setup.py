from setuptools import setup, find_packages

setup(
    name="nqd-audiomentations",
    version="1.1.3",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "audiomentations",
        "pyrubberband",
        "pyroomacoustics",
        "scipy",
        "librosa"
    ],
)