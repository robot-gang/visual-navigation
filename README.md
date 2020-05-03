# Visual Navigation in Dynamic Environments for Mobile Robots

The website: TODO

This is the final project of EECS C106B/206B | Robotic Manipulation and
Interaction course at UC Berkeley, Spring 2020.

# Members

- Son Tran
- Zuyong Li
- Yuanrong Han
- Shawn Shacterman

# How to develop this code?

## Setting things up

- Install miniconda/anaconda on your system
- Run the following command to set up the environment
    + `conda env create -f environment.yml`
- The environment's name is `visual_nav`
- To activate the environment: `conda activate visual_nav`
- To deactivate: `conda deactivate`

### Update environment.yml

If you want to install new packages for your code, you should follow the
following steps.

- Make sure your `~/.condarc` (conda configuration file, it is in your
  home folder) has these lines:
    + Make sure the channel `defaults` above the channel `conda-forge`
    the same order as the following lines

```.condarc
channels:
  - defaults
  - conda-forge
channel_priority: strict
```

- Pull the latest code from the repository: `git pull`
    + If there are conflicts, you should resolve them all before doing
    the next step.
- Run this following command (at the same directory with the
  environment.yml file) to get the new changes from other people
    + `conda env update -n visual_nav -f environment.yml`
- Search for the latest version of your package
    + e.g., `conda search numpy`
    + e.g., the latest of numpy is 1.18.1
- Install the new package with the first two parts of the version
    + e.g., `conda install numpy=1.18`
- Update the environment.yml (run the following command at the same
  directory with the environment.yml file)
    + `conda env export --from-history | grep -v "prefix" > environment.yml`
- Add and commit the change to master

## Coding

- Python version: 3.7
- Branches
    + master: the master branch
    + gh-pages: the website
- CODE, COMMIT, AND PUSH TO MASTER
    + CREATE A NEW FOLDER FOR YOUR CODE
    + Put an empty `__init__.py` in every folder that you created to
    make it as a Python module. We can import and use these folders as
    normal Python modules/packages.
- Python docstring format: reStructuredText
