import os
from unittest import TestCase

import pandas as pd

import nbnode.nbnode_trees as nbtree
from nbnode.io.pickle_open_dump import pickle_open_dump
from nbnode.simulation.FlowSimulationTree import FlowSimulationTreeDirichlet


class TestTreeMeans(TestCase):
    @classmethod
    def setUpClass(self) -> None:
        # https://docs.python.org/3/library/unittest.html#unittest.TestCase.setUpClass
        # setUpClass is only called once for the whole class in contrast to setUp which
        # is called before every test.

        import re

        from nbnode.testutil.helpers import find_dirname_above_currentfile

        TESTS_DIR = find_dirname_above_currentfile()

        cellmat = pd.read_csv(
            os.path.join(
                TESTS_DIR, "testdata", "flowcytometry", "gated_cells", "cellmat.csv"
            )
        )
        # FS TOF (against FS INT which is "FS")
        cellmat.rename(columns={"FS_TOF": "FS.0"}, inplace=True)
        cellmat.columns = [re.sub("_.*", "", x) for x in cellmat.columns]
        self.cellmat = cellmat

        celltree = nbtree.tree_complete_aligned_trunk()
        celltree.data = self.cellmat
        celltree.id_preds(celltree.predict())

        self.flowsim_trunk = FlowSimulationTreeDirichlet(
            node_percentages=None,
            rootnode=celltree,
            data_cellgroup_col=None,
            include_features="dataset_melanoma_short",
            verbose=True,
        )

    def test_TreeMeanDistributionSampler(self):
        from nbnode.simulation.TreeMeanDistributionSampler import (
            TreeMeanDistributionSampler,
        )

        print("test")
        os.makedirs("tests_output", exist_ok=True)
        pickle_open_dump(self.flowsim_trunk, "tests_output/flowsim_trunk.pickle")
        tmds = TreeMeanDistributionSampler(
            "tests_output/flowsim_trunk.pickle",
            population_name_to_change="/AllCells/DN",
            n_samples=2,
            n_cells=100,
        )
        result1 = tmds.sample()
        (
            all_true_popcounts,
            all_changed_parameters,
            all_sampled_samples,
            all_targets,
        ) = result1
        # That the number of cells is equal is not always the case but here the DP are
        # so strong.l
        assert all_true_popcounts.astype(int).equals(
            pd.DataFrame.from_dict(
                {
                    "sample_0": {
                        "/AllCells/CD4+/CD8-/naive": 2,
                        "/AllCells/CD4-/CD8+/naive": 0,
                        "/AllCells/DP": 98,
                    },
                    "sample_1": {
                        "/AllCells/CD4+/CD8-/naive": 2,
                        "/AllCells/CD4-/CD8+/naive": 0,
                        "/AllCells/DP": 98,
                    },
                }
            ).astype(int)
        )
        assert all_changed_parameters == []
        assert len(all_sampled_samples) == 2
        # assert all_targets == [1e-09, 0.018689260482788086]  # "interactively"
        # assert all_targets == [1e-09, 1e-09]  # "automatically"
        assert all(x.shape == (100, 13) for x in all_sampled_samples)

        tmds = TreeMeanDistributionSampler(
            flowsim_tree=self.flowsim_trunk,
            population_name_to_change="/AllCells/DN",
            n_samples=2,
            n_cells=100,
        )
        result2 = tmds.sample()
        assert result1[0].equals(result2[0])
        assert all([x.equals(y) for x, y in zip(result1[2], result2[2])])

        tmds = TreeMeanDistributionSampler(
            flowsim_tree=self.flowsim_trunk,
            population_name_to_change="/AllCells/DN",
            n_samples=2,
            n_cells=100,
            save_dir=None,
        )
        result3 = tmds.sample()
        assert result1[0].equals(result3[0])
        assert all([x.equals(y) for x, y in zip(result1[2], result3[2])])

        tmds = TreeMeanDistributionSampler(
            flowsim_tree=self.flowsim_trunk,
            population_name_to_change="/AllCells/DN",
            n_samples=2,
            n_cells=100,
            verbose=False,
        )
        result4 = tmds.sample()
        assert result1[0].equals(result4[0])
        assert all([x.equals(y) for x, y in zip(result1[2], result4[2])])

        tmds = TreeMeanDistributionSampler(
            flowsim_tree=self.flowsim_trunk,
            population_name_to_change="/AllCells/DN",
            n_samples=2,
            n_cells=100,
            verbose=True,
        )
        # The following is the default created directory where the generated samples are
        # stored. The directory is created by the TreeMeanDistributionSampler class.
        assert os.path.exists("sim/sim00_m0.sd1")

        tmds = TreeMeanDistributionSampler(
            "tests_output/flowsim_trunk.pickle",
            population_name_to_change="/AllCells/DN",
            n_samples=2,
            n_cells=100,
            save_changed_parameters=True,
        )
        result5 = tmds.sample()
        (
            all_true_popcounts,
            all_changed_parameters,
            all_sampled_samples,
            all_targets,
        ) = result5
        assert all_changed_parameters != []

    def test_TreeMeanRelative(self):
        from nbnode.simulation.TreeMeanRelative import TreeMeanRelative

        print("test")
        os.makedirs("tests_output", exist_ok=True)
        pickle_open_dump(self.flowsim_trunk, "tests_output/flowsim_trunk.pickle")
        tmds = TreeMeanRelative(
            "tests_output/flowsim_trunk.pickle",
            change_pop_mean_proportional={"/AllCells/DN": 1},
            n_samples=2,
            n_cells=100,
        )
        result1 = tmds.sample()
        tmds = TreeMeanRelative(
            flowsim_tree=self.flowsim_trunk,
            change_pop_mean_proportional={"/AllCells/DN": 1},
            n_samples=2,
            n_cells=100,
            save_dir=None,
        )
        result2 = tmds.sample()

        tmds = TreeMeanRelative(
            "tests_output/flowsim_trunk.pickle",
            change_pop_mean_proportional={"/AllCells/DN": 1},
            n_samples=2,
            n_cells=100,
            save_changed_parameters=True,
        )
        result3 = tmds.sample()
        (true_popcounts, changed_parameters, sampled_samples) = result3
        assert changed_parameters != []

        tmds = TreeMeanRelative(
            "tests_output/flowsim_trunk.pickle",
            change_pop_mean_proportional={"/AllCells/DN": 1},
            n_samples=2,
            n_cells=100,
            save_changed_parameters=False,
        )
        result4 = tmds.sample()
        (true_popcounts, changed_parameters, sampled_samples) = result4
        assert changed_parameters is None

        tmds = TreeMeanRelative(
            "tests_output/flowsim_trunk.pickle",
            change_pop_mean_proportional={"/AllCells/DN": 1},
            n_samples=2,
            n_cells=100,
            verbose=False,
        )
        tmds = TreeMeanRelative(
            "tests_output/flowsim_trunk.pickle",
            change_pop_mean_proportional={"/AllCells/DN": 1},
            n_samples=2,
            n_cells=100,
            verbose=True,
        )
        print(result1, result2, result3, result4)
