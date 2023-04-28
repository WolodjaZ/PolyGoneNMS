from setuptools import find_packages, setup


def read_version():
    with open("VERSION", "r") as f:
        return f.read().strip()


def read_requirements(path: str):
    with open(path, "r") as f:
        return f.read().splitlines()


def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()


setup(
    name="polygone-nms",
    version=read_version(),
    description=(
        "Efficient and distributed " "polygon Non-Maximum Suppression (NMS) library"
    ),
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="Vladimir Zaigrajew",
    author_email="vladimirzaigrajew@gmail.com",
    url="https://github.com/WolodjaZ/polygone-nms",
    packages=find_packages(exclude=["tests"]),
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "dev": read_requirements("requirements-dev.txt"),
        "docs": read_requirements("requirements-docs.txt"),
    },
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Utilities",
    ],
    python_requires=">=3.8",
)
