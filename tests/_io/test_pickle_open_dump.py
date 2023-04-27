import os

import nbnode_pyscaffold.nbnode_trees as nbtree
from nbnode_pyscaffold.testutil.helpers import find_dirname_above_currentfile

TESTS_DIR = find_dirname_above_currentfile()

from nbnode_pyscaffold.io.pickle_open_dump import pickle_open_dump


def test_pickle_open_dump():
    pickle_open_dump("test", "tests_output/test.pkl")


def test_pickle_open_dump_NBNode():
    import pandas as pd

    yternary = pd.read_csv(
        os.path.join(
            TESTS_DIR, "testdata", "flowcytometry", "gated_cells", "yternary.csv"
        )
    )
    celltree = nbtree.tree_complete_cell()
    yternary_preds = celltree.predict(values=yternary)
    celltree.id_preds(yternary_preds)
    # You have to set the data manually
    celltree.data = yternary
    assert len(celltree.data) == 999
    pickle_open_dump(celltree, "tests_output/test.pkl")
