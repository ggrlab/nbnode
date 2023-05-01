import os

import nbnode.nbnode_trees as nbtree
from nbnode.io.pickle_open_dump import pickle_open_dump
from nbnode.testutil.helpers import find_dirname_above_currentfile

TESTS_DIR = find_dirname_above_currentfile()


def test_pickle_open_dump():
    os.makedirs("tests_output/", exist_ok=True)
    pickle_open_dump("test", "tests_output/test.pkl")


def test_pickle_open_dump_NBNode():
    import pandas as pd

    os.makedirs("tests_output/", exist_ok=True)

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
