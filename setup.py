import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mly",
    version="0.6",
    author="Vasileios Skliris",
    author_email='vas.skliris@gmail.com',
    description='This tool helps you create training and testing data for ML to use for gravitational wave detection.',
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
        "tensorflow >= 2.6",
        "numpy",
        "scikit-learn",
        "pandas",
        "pytest",
        "pycondor",
        "pycbc",
        "healpy",
        "ligo.skymap"
    ]
)








