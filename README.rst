.. These are examples of badges you might want to add to your README:
   please update the URLs accordingly

    .. image:: https://api.cirrus-ci.com/github/<USER>/nbnode_pyscaffold.svg?branch=main
        :alt: Built Status
        :target: https://cirrus-ci.com/github/<USER>/nbnode_pyscaffold
    .. image:: https://readthedocs.org/projects/nbnode_pyscaffold/badge/?version=latest
        :alt: ReadTheDocs
        :target: https://nbnode_pyscaffold.readthedocs.io/en/stable/
    .. image:: https://img.shields.io/coveralls/github/<USER>/nbnode_pyscaffold/main.svg
        :alt: Coveralls
        :target: https://coveralls.io/r/<USER>/nbnode_pyscaffold
    .. image:: https://img.shields.io/pypi/v/nbnode_pyscaffold.svg
        :alt: PyPI-Server
        :target: https://pypi.org/project/nbnode_pyscaffold/
    .. image:: https://img.shields.io/conda/vn/conda-forge/nbnode_pyscaffold.svg
        :alt: Conda-Forge
        :target: https://anaconda.org/conda-forge/nbnode_pyscaffold
    .. image:: https://pepy.tech/badge/nbnode_pyscaffold/month
        :alt: Monthly Downloads
        :target: https://pepy.tech/project/nbnode_pyscaffold
    .. image:: https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter
        :alt: Twitter
        :target: https://twitter.com/nbnode_pyscaffold

.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/

|

=================
nbnode_pyscaffold
=================


    Add a short description here!


A longer description of your project goes here...


.. _pyscaffold-notes:

Note
====

This project has been set up using PyScaffold 4.4. For details and usage
information on PyScaffold see https://pyscaffold.org/::

    pip install pyscaffold
    putup nbnode_pyscaffold
    cd nbnode_pyscaffold
    # Create nbnode_pyscaffold within gitlab, without README
    git branch -m master main
    git remote add origin git@git.uni-regensburg.de:ccc_verse/nbnode_pyscaffold.git
    git push -u origin --all
    conda create -y -n nbnode_pyscaffold python=3.8
    conda activate nbnode_pyscaffold
    # Select nbnode_pyscaffold as default python interpreter in VsCode
    #   select a single python file, then on the bottom right, the current python interpreter name
    #   pops up. Click on it and select the "nbnode_pyscaffold" interpreter.
    # Make sure that the correct pip is used:
    #   Something like: /home/gugl/.conda_envs/nbnode_pyscaffold/bin/pip
    which pip
    pip install tox
    tox --help

    # Have a clean git, then add gitlab-ci with pyscaffold
    putup --update . --gitlab
    pip install pre-commit
    pre-commit autoupdate



Tests
====
For some tests you need data files, which are not included in the repository.
Especially all tests in `tests/specific_analyses` need data. 
You can obtain the data by running the following command in the root directory of the repository:

```
bash tests/specific_analyses/e01_download_intraassay.sh
```

