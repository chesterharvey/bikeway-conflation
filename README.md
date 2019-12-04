# Examining the Representativeness of OSM Bicycle Infrastructure in U.S. Cities

[![Build Status](https://travis-ci.com/ds421/chester-final-project.svg?token=Mygtej6PfhJ5XuYVyWtK&branch=master)](https://travis-ci.com/ds421/chester-final-project)  [![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/ds421/chester-final-project/master?filepath=Final%20Project.ipynb)

## About
This repo was developed by Chester Harvey as a final project for ESPM 288 at UC Berkeley.

## Proposal
The project proposal is presented in the `proposal.md` in the root directory. This document was compiled by the `proposal.ipynb` Jupyter Notebook, also in the root directory.

## Deliverables

Project deliverables are presented in several files:

- `Final Project.ipynb` : Jupyter Notebook containing all analyses and associated narrative. This is the native and preferred format for viewing project results. Run it live online with [Binder](https://mybinder.org/v2/gh/ds421/chester-final-project/master?filepath=Final%20Project.ipynb). 
- `Final Project.md` : Markdown version of `Final Project.ipynb`
- `compare_bikeways.py` : Module of project-specific Python scripts

## Special Files

### Project Resources

- `images`: Image resources referenced by Jupyter Notebooks and markdown documents
- `bikeway_shapefiles` : Shapefiles referenced by `Final Project.ipynb`
- `binder` : Files used to configure [Binder](https://mybinder.org/)

### Common Files
- `README.md`: This file; a general overview of the repository.
- `.gitignore`: Specifies files that will not be committed to GitHub.

### Testing Infrastructure
- `.travis.yml`: A configuration file for automatically running continuous integration checks to verify reproducibility.
- `REQUIREMENTS.txt`: A metadata file listing Python packages required for project scripts to run.
- `tests/test.py`: A Python script that runs Jupyter Notebooks to ensure they are fully executable. Currently, this script only runs the `proposal.ipynb` notebook. It will be updated to recursivly identify and test all notebooks within the respository.
