from setuptools import setup, find_packages

setup(
    name="nqd_audiomentations",
    version="1.0.1",
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