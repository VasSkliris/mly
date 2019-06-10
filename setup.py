import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mly",
    version="0.0.20",
    author="Vasileios Skliris",
    author_email='vas.skliris@gmail.com',
    description='Dataset generation and tools for ML in gravitational waves',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='http://pypi.python.org/pypi/mly/',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "gwpy >= 0.13.1",
        "Keras >= 2.2.4",
        "tensorflow >= 1.12.0",
        "numpy >= 1.16.1",
        "sklearn"
    ]
)








