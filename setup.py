from setuptools import setup, find_packages

setup(
    name="nqd_audiomentations",
    version="1.0.3",
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