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
nbnode
=================


    Add a short description here!


A longer description of your project goes here...


.. _pyscaffold-notes:

Note
====

This project has been set up using PyScaffold 4.4. For details and usage
information on PyScaffold see https://pyscaffold.org/::

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


Getting started
====
Optional: Create a (conda) environment and activate it.

.. code-block:: bash

    conda create -y -n conda_nbnode python=3.8
    conda activate conda_nbnode

Install and use nbnode install it from source. 
.. code-block:: bash

    git clone https://github.com/ggrlab/nbnode
    cd nbnode
    pip install --upgrade pip
    pip install . 

Base-functionality of the package is to enable non-binary trees. The following creates
a tree with a root node ``a`` and three children ``a0``, ``a1`` and ``a2``. ``a1`` is the only child with another child ``a1a``.

.. code-block::

    a
    ├── a0
    ├── a1
    │   └── a1a
    └── a2

A basic non-binary node (``NBNode``) consists of four important attributes:

    - ``name`` The name of the node. This is the only mandatory attribute.
    - ``parent`` The parent node of this node.
    - ``decision_name`` The name of the value leading to this node. 
    - ``decision_value`` The value leading to this node.

The name of the node must only be unique within all childs of the parent node.
The ``decision_name`` and ``decision_value`` are the named values leading to this node. Note that 
``decision_name`` must be a string, but ``decision_value`` can be anything, including strings, integers, floats, etc.

To build the tree above, we can use the following code:


.. code-block:: python
    
    from nbnode import NBNode
    # Create the root node "a"
    mytree = NBNode("a")
    # Create the node "a0" which 
    #  - Is a child of "mytree" 
    #  - Has the decision_name "m1" 
    #  - Has the decision_value -1
    NBNode("a0", parent=mytree, decision_value=-1, decision_name="m1")

    a1 = NBNode("a1", parent=mytree, decision_value=1, decision_name="m1")
    
    # Create the node "a2" which 
    #  - Is a child of "mytree" 
    #  - Has the decision_name "m3" 
    #  - Has the decision_value "another"
    NBNode("a2", parent=mytree, decision_value="another", decision_name="m3")
    NBNode("a1a", parent=a1, decision_value="test", decision_name="m2")

We can check if the previous tree was built correctly: 

.. code-block:: python

    mytree.pretty_print("__long__")
    #    a (counter:0, decision_name:None, decision_value:None)
    #    ├── a0 (counter:0, decision_name:m1, decision_value:-1)
    #    ├── a1 (counter:0, decision_name:m1, decision_value:1)
    #    │   └── a1a (counter:0, decision_name:m2, decision_value:test)
    #    └── a2 (counter:0, decision_name:m3, decision_value:another)

Finally, we use the tree to predict the final node of a new data point.
The following values, supplied as two lists ``values`` and ``names`` are used to predict the final node.

.. code-block:: python

    single_prediction = mytree.predict(
        values=[1, "test", 2], names=["m1", "m2", "m3"]
    )
    print(single_prediction)


Tutorials 
====
.. * [Part 1 - Non-binary node ](https://github.com/whitews/FlowKit/blob/master/docs/notebooks/flowkit-tutorial-part01-sample-class.ipynb)
.. * [Part 2 - transforms Module & Matrix Class](https://github.com/whitews/FlowKit/blob/master/docs/notebooks/flowkit-tutorial-part02-transforms-module-matrix-class.ipynb)
.. * [Part 3 - GatingStrategy & GatingResults Classes](https://github.com/whitews/FlowKit/blob/master/docs/notebooks/flowkit-tutorial-part03-gating-strategy-and-gating-results-classes.ipynb)
.. * [Part 4 - gates Module](https://github.com/whitews/FlowKit/blob/master/docs/notebooks/flowkit-tutorial-part04-gates-module.ipynb)
.. * [Part 5 - Session Class](https://github.com/whitews/FlowKit/blob/master/docs/notebooks/flowkit-tutorial-part05-session-class.ipynb)
.. * [Part 6 - Workspace Class](https://github.com/whitews/FlowKit/blob/master/docs/notebooks/flowkit-tutorial-part06-workspace-class.ipynb)


Tests
====
For some tests you need data files, which are not included in the repository.
Especially all tests in `tests/specific_analyses` need data.
You can obtain the data by downloading the data from zenodo: 

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.7883353.svg
   :target: https://doi.org/10.5281/zenodo.7883353

.. code-block:: bash 
    pip install requests
    python tests/specific_analyses/e02_download_intraassay_zenodo.py


