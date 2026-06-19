"""Setup script for the FedFOD package."""

import os
from setuptools import setup, find_packages


def read_requirements(filepath: str = "requirements.txt") -> list[str]:
    """Read requirements from file, filtering out git+ lines and comments."""
    requirements = []
    if not os.path.isfile(filepath):
        return requirements
    with open(filepath, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            # Skip empty lines, comments, and git+ direct references
            if not line or line.startswith("#") or line.startswith("git+"):
                continue
            requirements.append(line)
    return requirements


def read_long_description() -> str:
    """Read the long description from README.md if it exists."""
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.isfile(readme_path):
        with open(readme_path, "r", encoding="utf-8") as fh:
            return fh.read()
    return ""


setup(
    name="fedfod",
    version="1.0.0",
    author="FedFOD Research Team",
    author_email="fedfod-research@example.org",
    description="Federated Foreign Object Debris Detection",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/fedfod-research/fedfod",
    project_urls={
        "Bug Tracker": "https://github.com/fedfod-research/fedfod/issues",
        "Documentation": "https://github.com/fedfod-research/fedfod/wiki",
    },
    packages=find_packages(exclude=["tests", "tests.*", "scripts", "notebooks"]),
    install_requires=read_requirements(),
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Operating System :: OS Independent",
    ],
    keywords=[
        "federated-learning",
        "object-detection",
        "foreign-object-debris",
        "airport-safety",
        "RT-DETR",
        "SCAFFOLD",
        "privacy-preserving",
        "SMPC",
    ],
    entry_points={
        "console_scripts": [
            "fedfod-train=fedfod.train:main",
            "fedfod-server=fedfod.server:main",
            "fedfod-client=fedfod.client:main",
        ],
    },
    zip_safe=False,
)
