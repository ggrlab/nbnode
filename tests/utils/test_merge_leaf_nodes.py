import os
from unittest import TestCase

import numpy as np
import pandas as pd

import nbnode.nbnode_trees as nbtree
from nbnode.simulation.FlowSimulationTree import FlowSimulationTreeDirichlet
from nbnode.utils.merge_leaf_nodes import merge_leaf_nodes


class TestMergeLeafNodes(TestCase):
    @classmethod
    def setUpClass(self) -> None:
        from nbnode.testutil.helpers import find_dirname_above_currentfile

        self.TESTS_DIR = find_dirname_above_currentfile()
        # https://docs.python.org/3/library/unittest.html#unittest.TestCase.setUpClass
        # setUpClass is only called once for the whole class in contrast to setUp which
        # is called before every test.

        import re

        cellmat = pd.read_csv(
            os.path.join(
                self.TESTS_DIR,
                "testdata",
                "flowcytometry",
                "gated_cells",
                "cellmat.csv",
            )
        )
        # FS TOF (against FS INT which is "FS")
        cellmat.rename(columns={"FS_TOF": "FS.0"}, inplace=True)
        cellmat.columns = [re.sub("_.*", "", x) for x in cellmat.columns]
        self.cellmat = cellmat

    def test_merge_leaf_nodes(self):
        celltree = nbtree.tree_complete_aligned()
        celltree.data = self.cellmat
        celltree.id_preds(celltree.predict())

        flowsim = FlowSimulationTreeDirichlet(
            node_percentages=None,
            rootnode=celltree,
            data_cellgroup_col=None,
            include_features="dataset_melanoma_short",
            verbose=True,
        )
        # Usually there would be much more leafs here, but we only use a small test
        print(flowsim.mean_leafs)
        # /AllCells/CD4+/CD8-/naive/CD27+/CD28+/CD57+/PD1+    0.018227
        # /AllCells/CD4+/CD8-/naive/CD27-/CD28+/CD57+/PD1+    0.006222
        # /AllCells/CD4-/CD8+/naive/CD27+/CD28-/CD57+/PD1+    0.005207
        # /AllCells/DP                                        0.970345

        assert np.isclose(
            merge_leaf_nodes(flowsim.mean_leafs, "/AllCells/CD4-/CD8+/naive"),
            0.005207,
            atol=1e-5,
        )
        assert np.isclose(
            merge_leaf_nodes(flowsim.mean_leafs, "/AllCells/CD4+/CD8-/naive"),
            0.018227 + 0.006222,
            atol=1e-5,
        )
        assert np.isclose(
            merge_leaf_nodes(flowsim.mean_leafs, "/AllCells"),
            1,
            atol=1e-5,
        )
