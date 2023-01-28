import setuptools



setuptools.setup(
    name="stgp", 
    version="0.0.1",
    author="O Hamelijnck",
    author_email="ohamelijnck@turing.ac.uk",
    description="SpaTial GP (STGP) library in jax",
    long_description="",
    long_description_content_type="text/markdown",
    url="N/A",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires = [
        "jax",
        "objax@git+https://github.com/google/objax.git#egg=sacred",
        "batchjax@git+https://github.com/defaultobject/batchjax#egg=sacred",
        "chex",
        "tqdm",
        "black",
        "flake8",
        "flake8-docstrings",
        "bibtexparser",
        "pandas",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "pytest"
    ],
    extras_require={
        'm1': [
            'tensorflow-macos',
            'tensorflow_probability',
        ],
        'other': [
            'tensorflow',
            'tensorflow_probability',
        ]
    }
)
