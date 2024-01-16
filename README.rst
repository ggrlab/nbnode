.. These are examples of badges you might want to add to your README:
   please update the URLs accordingly

    .. image:: https://api.cirrus-ci.com/github/<USER>/nbnode.svg?branch=main
        :alt: Built Status
        :target: https://cirrus-ci.com/github/<USER>/nbnode
    .. image:: https://readthedocs.org/projects/nbnode/badge/?version=latest
        :alt: ReadTheDocs
        :target: https://nbnode.readthedocs.io/en/stable/
    .. image:: https://img.shields.io/coveralls/github/<USER>/nbnode/main.svg
        :alt: Coveralls
        :target: https://coveralls.io/r/<USER>/nbnode
    .. image:: https://img.shields.io/pypi/v/nbnode.svg
        :alt: PyPI-Server
        :target: https://pypi.org/project/nbnode/
    .. image:: https://img.shields.io/conda/vn/conda-forge/nbnode.svg
        :alt: Conda-Forge
        :target: https://anaconda.org/conda-forge/nbnode
    .. image:: https://pepy.tech/badge/nbnode/month
        :alt: Monthly Downloads
        :target: https://pepy.tech/project/nbnode
    .. image:: https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter
        :alt: Twitter
        :target: https://twitter.com/nbnode

.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/

|

=================
NBNode
=================


    A python package for non-binary trees and simulation of flow cytometry data.


NBNode enables non-binary decision trees with multiple decisions at each node.
Additionally it enables a dirichlet distribution based simulation of flow cytometry data.

Installation
============

.. code-block:: bash

    conda create -y -n conda_nbnode python=3.8
    conda activate conda_nbnode
    git clone https://git.uni-regensburg.de/ccc_verse/nbnode
    cd nbnode
    pip install --upgrade pip
    pip install .


Tutorials
=========

See the documentation and the (within) presented jupyter notebooks https://ggrlab.github.io/nbnode/


Package setup
=============

This project has been set up using PyScaffold 4.4. For details and usage
information on PyScaffold see https://pyscaffold.org/.

.. code-block:: bash

    pip install pyscaffold
    putup nbnode
    cd nbnode
    # Create nbnode within gitlab, without README
    git branch -m master main
    git remote add origin git@git.uni-regensburg.de:ccc_verse/nbnode.git
    git push -u origin --all
    conda create -y -n conda_nbnode python=3.8
    conda activate conda_nbnode
    # Select conda_nbnode as default python interpreter in VsCode
    #   select a single python file, then on the bottom right, the current python interpreter name
    #   pops up. Click on it and select the "conda_nbnode" interpreter.
    # Make sure that the correct pip is used:
    #   Something like: /home/gugl/.conda_envs/conda_nbnode/bin/pip
    which pip
    pip install tox
    tox --help

    # Have a clean git, then add gitlab-ci with pyscaffold
    putup --update . --gitlab
    pip install pre-commit
    pre-commit autoupdate

    # Migration to github:

        putup --update . --github

    # pypi:
    # login, then create a new token https://pypi.org/manage/account/token/
    # re-comment (after I commented them out)
    # .github/workflows/ci.yml:
    #       publish: [...]





Tests
========
For some tests you need data files, which are not included in the repository.
Especially all tests in `tests/specific_analyses` need data.
You can obtain the data by downloading the data from zenodo:

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.7890571.svg
   :target: https://doi.org/10.5281/zenodo.7890571

.. code-block:: bash

    pip install requests
    python tests/specific_analyses/e02_download_intraassay_zenodo.py
