import setuptools



setuptools.setup(
    name="gpjax", 
    version="0.0.1",
    author="O Hamelijnck, W Wilkinson",
    author_email="ohamelijnck@turing.ac.uk,william.wilkinson@aalto.fi",
    description="GP library in jax",
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
)
