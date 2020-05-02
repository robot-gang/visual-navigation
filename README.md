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

```.condarc
channels:
  - defaults
  - conda-forge
channel_priority: strict
```

- Install the new package
    + e.g., `conda install numpy`
- Update the environment.yml
    + `conda env export --from-history > environment.yml`
    + This will create a merge conflict later with other versions of
    environment.yml on other branches, but we will resolve it later when
    we merge to master.
- Add and commit the change to your branch

## Coding

- Branches
    + master: the master branch, DON'T COMMIT YOUR CODE HERE!
    + gh-pages: the website
    + <task_name>: create a new branch for your task
        * e.g., moving_object, tcn, etc.
- Code, commit, and push to your branch
    + Create a new folder for your code
    + Put an empty `__init__.py` in every folders that you created to
    make it as a Python module. We can import and use these folder as
    normal Python module/package.
- We will merge other branches to the master later.
