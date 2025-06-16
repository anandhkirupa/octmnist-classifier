from setuptools import setup, find_packages
from pathlib import Path

setup(
    name='octmnist-classifier',
    version='0.1.2',
    packages=find_packages(exclude=["tests", "notebooks", "docs"]),
    install_requires=[
        'torch>=1.9.0',
        'numpy>=1.19',
        'matplotlib>=3.3',
        'scikit-learn>=0.24',
        'medmnist',
        'seaborn>=0.11',
        'pillow>=8.0',            
        'clustimage>=1.2.6',     
        'imbalanced-learn>=0.8',
    ],
    entry_points={
        'console_scripts': [
            'octmnist-predict=octmnist_classifier.cli:main'
        ]
    },
    include_package_data=True,
    author='Kirupanandan Jagadeesan',
    author_email='kirupana@buffalo.edu',
    description='A CNN-based classifier for OCTMNIST retinal disease classification with training, prediction, and CLI tools.',
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Intended Audience :: Science/Research'
    ],
    python_requires='>=3.7',
)
