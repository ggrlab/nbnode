import copy
import os
from unittest import TestCase

import anytree
import numpy as np
import pandas as pd

import nbnode.nbnode_trees as nbtree
from nbnode.nbnode import NBNode
from nbnode.simulation.FlowSimulationTree import FlowSimulationTreeDirichlet
from nbnode.testutil.helpers import find_dirname_above_currentfile

TESTS_DIR = find_dirname_above_currentfile()


class TestFlowSimulation(TestCase):
    @classmethod
    def setUpClass(self) -> None:
        # https://docs.python.org/3/library/unittest.html#unittest.TestCase.setUpClass
        # setUpClass is only called once for the whole class in contrast to setUp which
        # is called before every test.

        import re

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
            # With data_cellgroup_col=None,
            # all cells are assumed to come from the same sample
            data_cellgroup_col=None,
            include_features="dataset_melanoma_short",
            verbose=True,
        )

    def test_tree_simple_predict(self):
        mytree = nbtree.tree_simple()

        single_prediction = mytree.predict(
            values=[1, "test", 2], names=["m1", "m2", "m3"]
        )
        assert [x.name for x in single_prediction.iter_path_reverse()] == [
            "a1a",
            "a1",
            "a",
        ]

        single_prediction = mytree.predict(
            values=[-1, "test", 2], names=["m1", "m2", "m3"]
        )
        assert [x.name for x in single_prediction.iter_path_reverse()] == ["a0", "a"]

    def test_FlowSimulationTreeDirichlet_init(self):
        yternary = pd.read_csv(
            os.path.join(
                TESTS_DIR, "testdata", "flowcytometry", "gated_cells", "yternary.csv"
            )
        )
        celltree = nbtree.tree_complex()
        yternary_preds = celltree.predict(values=yternary)
        celltree.id_preds(yternary_preds)

        yternary["fake_activations"] = np.random.default_rng(12936).random(
            len(yternary)
        )
        celltree.data = yternary
        flowsim = FlowSimulationTreeDirichlet(
            rootnode=celltree,
            include_features=["CD57", "PD1", "fake_activations"],
            node_percentages=None,
            data_cellgroup_col="CD8",  # JUST FOR TESTING!
        )
        print(flowsim)

    def test_FlowSimulationTreeDirichlet(self):
        yternary = pd.read_csv(
            os.path.join(
                TESTS_DIR, "testdata", "flowcytometry", "gated_cells", "yternary.csv"
            )
        )
        celltree = nbtree.tree_complex()
        yternary_preds = celltree.predict(values=yternary)
        celltree.id_preds(yternary_preds)

        yternary["fake_activations"] = np.random.default_rng(12936).random(
            len(yternary)
        )
        celltree.data = yternary

        with self.assertRaises(KeyError):
            FlowSimulationTreeDirichlet(
                include_features=["CD57", "PD1", "fake_activations"],
                node_percentages=None,
                rootnode=celltree,
            )

        flowsim = FlowSimulationTreeDirichlet(
            rootnode=celltree,
            include_features=["CD57", "PD1", "fake_activations"],
            node_percentages=None,
            data_cellgroup_col="CD8",  # JUST FOR TESTING!
        )
        # test sampling a single sample
        flowsim.set_seed(1230847)
        x = flowsim.sample(n_cells=1)
        print(x.iloc[0].to_numpy())
        assert all(
            np.isclose(
                x.iloc[0].to_numpy(), np.array([0.9913707, -0.78346626, 0.07124489])
            )
        )
        x = flowsim.sample(n_cells=100)
        assert not all(
            np.isclose(
                x.iloc[0].to_numpy(), np.array([0.9913707, -0.78346626, 0.07124489])
            )
        )
        flowsim.set_seed(1230847)
        x = flowsim.sample(n_cells=1)
        assert all(
            np.isclose(
                x.iloc[0].to_numpy(), np.array([0.9913707, -0.78346626, 0.07124489])
            )
        )
        assert x.shape[0] == 1
        assert x.shape[1] == 3
        flowsim.set_seed(1230847)
        x = flowsim.sample(n_cells=100)
        print(x.iloc[0])
        assert all(
            np.isclose(x.iloc[99].to_numpy(), np.array([0.0, 0.0, 0.20777241447137335]))
        )
        assert x.shape[0] == 100
        assert x.shape[1] == 3

    def test_FlowSimulationTreeDirichlet_singlesample(self):
        yternary = pd.read_csv(
            os.path.join(
                TESTS_DIR, "testdata", "flowcytometry", "gated_cells", "yternary.csv"
            )
        )
        celltree = nbtree.tree_complex()
        yternary_preds = celltree.predict(values=yternary)
        celltree.id_preds(yternary_preds)

        yternary["fake_activations"] = np.random.default_rng(12936).random(
            len(yternary)
        )
        celltree.data = yternary
        flowsim = FlowSimulationTreeDirichlet(
            rootnode=celltree,
            include_features=["CD57", "PD1", "fake_activations"],
            node_percentages=None,
            # With data_cellgroup_col=None,
            # all cells are assumed to come from the same sample
            data_cellgroup_col=None,
        )

        x = flowsim.sample(n_cells=100)
        assert x.shape[0] == 100
        assert x.shape[1] == 3

    def test_rng_numpy(self):
        a = np.random.default_rng(120983)
        newnumber = a.random(1)
        assert np.isclose(newnumber, 0.71607698)
        assert not np.isclose(newnumber, a.random(1))
        a = np.random.default_rng(120983)
        assert np.isclose(0.71607698, a.random(1))

    def test_FlowSimulationTreeDirichlet_2(self):
        yternary = pd.read_csv(
            os.path.join(
                TESTS_DIR, "testdata", "flowcytometry", "gated_cells", "yternary.csv"
            )
        )
        celltree = nbtree.tree_complex()
        yternary_preds = celltree.predict(values=yternary)
        celltree.id_preds(yternary_preds)

        yternary["fake_activations"] = np.random.default_rng(12936).random(
            len(yternary)
        )
        celltree.data = yternary

        leaf_nodes = [x for x in anytree.PreOrderIter(celltree) if x.is_leaf]

        flowsim = FlowSimulationTreeDirichlet(
            rootnode=celltree,
            include_features=["CD57", "PD1", "fake_activations"],
            node_percentages=None,
            # With data_cellgroup_col=None,
            # all cells are assumed to come from the same sample
            data_cellgroup_col=None,
        )

        flowsim.set_seed(10289)
        z = flowsim.sample(
            n_cells=100,
            # mean=flowsim.population_distribution_parameters["mean"] * 2,
            # cov=np.diag(np.diag(flowsim.population_distribution_parameters["cov"])),
            # population_names=flowsim.population_distribution_parameters["mean"].index,
        )
        print(leaf_nodes, z)

    def test_subset_by_nodename(self):
        mytree = nbtree.tree_simple()
        for node_x in anytree.PreOrderIter(mytree):
            node_x: NBNode
            a = mytree[node_x.get_name_full()]
            print(a)
        assert mytree["nonexisting_node_name"] is None
        with self.assertRaises(ValueError):
            mytree["illegal_two_elements", "illegal_two_elements"]

    def test_FlowSimulationTreeDirichlet_return_sampled_cell_numbers(self):
        mytree = nbtree.tree_simpleB()
        a1a_values = {"values": [1, "test", None], "names": ["m1", "m2", "m3"]}
        a1b_values = {"values": [1, "tmp", None], "names": ["m1", "m2", "m3"]}
        samples = [a1a_values] * 3 + [a1b_values] * 2
        samples_predicted_nodes = [
            mytree.predict(**sample_values) for sample_values in samples
        ]
        mytree.id_preds(samples_predicted_nodes)
        mytree.count(samples_predicted_nodes, use_ids=True)
        mytree.pretty_print()

        mytree.data = pd.DataFrame(
            np.concatenate(
                [
                    # 3 samples, 2 features
                    np.zeros(shape=(3, 2)),
                    # 2 samples, 2 features
                    np.ones(shape=(2, 2)),
                ]
            ),
            columns=["f1", "f2"],
        )

        flowsim = FlowSimulationTreeDirichlet(
            rootnode=mytree,
            include_features=["f1", "f2"],
            node_percentages=None,
            # With data_cellgroup_col=None,
            # all cells are assumed to come from the same sample
            data_cellgroup_col=None,
        )
        flowsim.set_seed(1234)
        leaf_sample = flowsim.sample(10, return_sampled_cell_numbers=True)
        leaf_sample_popN = leaf_sample[1]
        assert len(leaf_sample_popN) == 2

    def test_FlowSimulationTree_dirichlet(self):
        yternary = pd.read_csv(
            os.path.join(
                TESTS_DIR, "testdata", "flowcytometry", "gated_cells", "yternary.csv"
            )
        )
        celltree = nbtree.tree_complex()
        yternary_preds = celltree.predict(values=yternary)
        celltree.id_preds(yternary_preds)

        yternary["fake_activations"] = np.random.default_rng(12936).random(
            len(yternary)
        )
        celltree.data = yternary

        flowsim = FlowSimulationTreeDirichlet(
            include_features=["CD57", "PD1", "fake_activations"],
            node_percentages=None,
            rootnode=celltree,
            # With data_cellgroup_col=None,
            # all cells are assumed to come from the same sample
            data_cellgroup_col=None,
        )

        flowsim.set_seed(10289)
        flowsim.sample(n_cells=100)

    def test_FlowSimulationTree_report_mean_precision(self):
        yternary = pd.read_csv(
            os.path.join(
                TESTS_DIR, "testdata", "flowcytometry", "gated_cells", "yternary.csv"
            )
        )
        celltree = nbtree.tree_complex()
        yternary_preds = celltree.predict(values=yternary)
        celltree.id_preds(yternary_preds)

        yternary["fake_activations"] = np.random.default_rng(12936).random(
            len(yternary)
        )
        celltree.data = yternary

        flowsim = FlowSimulationTreeDirichlet(
            include_features=["CD57", "PD1", "fake_activations"],
            node_percentages=None,
            rootnode=celltree,
            # With data_cellgroup_col=None,
            # all cells are assumed to come from the same sample
            data_cellgroup_col=None,
        )
        print(flowsim.precision)
        print(flowsim.mean_leafs)
        assert flowsim.pop_alpha("/AllCells") == flowsim.precision
        assert np.isclose(
            flowsim.pop_alpha("/AllCells/CD45+"),
            (
                flowsim.precision
                - flowsim.population_parameters["alpha"]["/AllCells/not CD45"]
            ),
        )
        assert flowsim.pop_mean("/AllCells") == 1
        assert np.isclose(
            flowsim.pop_mean("/AllCells/CD45+"),
            (
                flowsim.precision
                - flowsim.population_parameters["alpha"]["/AllCells/not CD45"]
            )
            / flowsim.precision,
        )

    def test_FlowSimulationTree_report_all_alpha(self):
        yternary = pd.read_csv(
            os.path.join(
                TESTS_DIR, "testdata", "flowcytometry", "gated_cells", "yternary.csv"
            )
        )
        celltree = nbtree.tree_complex()
        yternary_preds = celltree.predict(values=yternary)
        celltree.id_preds(yternary_preds)

        yternary["fake_activations"] = np.random.default_rng(12936).random(
            len(yternary)
        )
        celltree.data = yternary

        flowsim = FlowSimulationTreeDirichlet(
            include_features=["CD57", "PD1", "fake_activations"],
            node_percentages=None,
            rootnode=celltree,
            # With data_cellgroup_col=None,
            # all cells are assumed to come from the same sample
            data_cellgroup_col=None,
        )
        print(flowsim.precision)
        print(flowsim.mean_leafs)
        print(flowsim.alpha_all)

    def test_FlowSimulationTree_new_mean(self):
        yternary = pd.read_csv(
            os.path.join(
                TESTS_DIR, "testdata", "flowcytometry", "gated_cells", "yternary.csv"
            )
        )
        celltree = nbtree.tree_complex()
        yternary_preds = celltree.predict(values=yternary)
        celltree.id_preds(yternary_preds)

        yternary["fake_activations"] = np.random.default_rng(12936).random(
            len(yternary)
        )
        celltree.data = yternary

        flowsim = FlowSimulationTreeDirichlet(
            include_features=["CD57", "PD1", "fake_activations"],
            node_percentages=None,
            rootnode=celltree,
            # With data_cellgroup_col=None,
            # all cells are assumed to come from the same sample
            data_cellgroup_col=None,
        )
        # print(flowsim.precision)
        assert len(flowsim.pop_leafnode_names("/AllCells")) == len(
            [x for x in anytree.PreOrderIter(celltree) if x.is_leaf]
        )
        assert len(flowsim.pop_leafnode_names("/AllCells/not CD45")) == 1
        assert (
            len(flowsim.pop_leafnode_names("/AllCells/CD45+"))
            == len([x for x in anytree.PreOrderIter(celltree) if x.is_leaf]) - 1
        )
        assert len(flowsim.pop_leafnode_names("/AllCells/CD45+/CD3+/CD4+/CD8-")) == 1

        with self.assertRaises(ValueError):
            flowsim.new_pop_mean("/AllCells/not CD45", percentage=1.1)
        with self.assertRaises(ValueError):
            flowsim.new_pop_mean("/AllCells/not CD45", percentage=-0.1)

        flowsim.new_pop_mean("/AllCells/not CD45", percentage=0.1)
        assert np.isclose(flowsim.pop_mean("/AllCells/not CD45"), 0.1)

        flowsim.new_pop_mean("/AllCells/CD45+/CD3+", percentage=0.1)
        assert np.isclose(flowsim.pop_mean("/AllCells/CD45+/CD3+"), 0.1)

    def test_FlowSimulationTree_report_parameters(self):
        yternary = pd.read_csv(
            os.path.join(
                TESTS_DIR, "testdata", "flowcytometry", "gated_cells", "yternary.csv"
            )
        )
        celltree = nbtree.tree_complex()
        yternary_preds = celltree.predict(values=yternary)
        celltree.id_preds(yternary_preds)

        yternary["fake_activations"] = np.random.default_rng(12936).random(
            len(yternary)
        )
        celltree.data = yternary

        flowsim = FlowSimulationTreeDirichlet(
            include_features=["CD57", "PD1", "fake_activations"],
            node_percentages=None,
            rootnode=celltree,
            # With data_cellgroup_col=None,
            # all cells are assumed to come from the same sample
            data_cellgroup_col=None,
        )
        for x in anytree.PreOrderIter(celltree):
            print(
                x.get_name_full(),
                flowsim.pop_alpha(x.get_name_full()),
                flowsim.pop_mean(x.get_name_full()),
            )

    def test_FlowSimulationTree_reset_populations(self):
        yternary = pd.read_csv(
            os.path.join(
                TESTS_DIR, "testdata", "flowcytometry", "gated_cells", "yternary.csv"
            )
        )
        celltree = nbtree.tree_complex()
        yternary_preds = celltree.predict(values=yternary)
        celltree.id_preds(yternary_preds)

        yternary["fake_activations"] = np.random.default_rng(12936).random(
            len(yternary)
        )
        celltree.data = yternary

        flowsim = FlowSimulationTreeDirichlet(
            include_features=["CD57", "PD1", "fake_activations"],
            node_percentages=None,
            rootnode=celltree,
            # With data_cellgroup_col=None,
            # all cells are assumed to come from the same sample
            data_cellgroup_col=None,
        )
        assert np.isclose(flowsim.pop_mean("/AllCells/not CD45"), 0.28343526674499486)
        flowsim.reset_populations()
        assert np.isclose(flowsim.pop_mean("/AllCells/not CD45"), 0.28343526674499486)

        flowsim.new_pop_mean("/AllCells/not CD45", percentage=0.1)
        assert np.isclose(flowsim.pop_mean("/AllCells/not CD45"), 0.1)

        flowsim.reset_populations()
        assert np.isclose(flowsim.pop_mean("/AllCells/not CD45"), 0.28343526674499486)

    def test_FlowSimulationTree_nodata_noids(self):
        celltree = nbtree.tree_complete_aligned()

        with self.assertRaises(ValueError):
            # ValueError: rootnode.data is None.
            # Please set rootnode.data before creating the simulation
            flowsim = FlowSimulationTreeDirichlet(
                node_percentages=None,
                rootnode=celltree,
                # With data_cellgroup_col=None,
                # all cells are assumed to come from the same sample
                data_cellgroup_col=None,
                include_features="dataset_melanoma_short",
            )

        celltree.data = self.cellmat
        with self.assertRaises(ValueError):
            # ValueError: rootnode.ids does not contain any id.
            # In this simulation we assume that all cells originate from the root node,
            # therefore rootnode.ids must contain at least one id.
            # Please set rootnode.ids before creating the simulation. Usually:
            #
            #     predicted_nodes_per_cell = celltree.predict(cellmat)
            #     celltree.id_preds(predicted_nodes_per_cell)

            flowsim = FlowSimulationTreeDirichlet(
                node_percentages=None,
                rootnode=celltree,
                # With data_cellgroup_col=None,
                # all cells are assumed to come from the same sample
                data_cellgroup_col=None,
                include_features="dataset_melanoma_short",
            )

        # The following two commands are equivalent after celltree.data was set
        # predicted_nodes_per_cell = celltree.predict(celltree.data)
        predicted_nodes_per_cell = celltree.predict()
        celltree.id_preds(predicted_nodes_per_cell)
        flowsim = FlowSimulationTreeDirichlet(
            node_percentages=None,
            rootnode=celltree,
            # With data_cellgroup_col=None,
            # all cells are assumed to come from the same sample
            data_cellgroup_col=None,
            include_features="dataset_melanoma_short",
        )
        print(flowsim.sample(n_cells=10))

    def test_FlowSimulationTree_singleNode(self):
        celltree = nbtree.tree_complete_aligned_trunk()
        celltree.data = self.cellmat
        celltree.id_preds(celltree.predict())
        # celltree.count(use_ids=True)

        FlowSimulationTreeDirichlet(
            node_percentages=None,
            rootnode=celltree,
            # With data_cellgroup_col=None,
            # all cells are assumed to come from the same sample
            data_cellgroup_col=None,
            include_features="dataset_melanoma_short",
        )

        # celltree["/AllCells/DP"] is a single node without children
        with self.assertWarns(UserWarning):
            # Only one single leaf node found, all cells will be simulated from single
            # node, are you sure that is what you want?
            # The dirichlet parameter will be 1, only the estimatedcell_distributions
            # might make sense.
            FlowSimulationTreeDirichlet(
                node_percentages=None,
                rootnode=celltree["/AllCells/DP"],
                # With data_cellgroup_col=None,
                # all cells are assumed to come from the same sample
                data_cellgroup_col=None,
                include_features="dataset_melanoma_short",
            )

    def test_FlowSimulationTree_percentages(self):
        celltree = nbtree.tree_complete_aligned_trunk()
        celltree.data = self.cellmat
        celltree.id_preds(celltree.predict())

        leaf_nodes_data = [
            node for node in anytree.PreOrderIter(celltree) if node.is_leaf
        ]
        ncells_per_sample = celltree.data.shape[0]
        ncells_per_node_per_sample = pd.DataFrame(
            {"single_sample": {x.get_name_full(): len(x.data) for x in leaf_nodes_data}}
        )
        #                            single_sample
        # /AllCells/CD4+/CD8-/Tcm                0
        # /AllCells/CD4+/CD8-/Tem                0
        # /AllCells/CD4+/CD8-/Temra              1
        # /AllCells/CD4+/CD8-/naive             20
        # /AllCells/CD4-/CD8+/Tcm                0
        # /AllCells/CD4-/CD8+/Tem                0
        # /AllCells/CD4-/CD8+/Temra              0
        # /AllCells/CD4-/CD8+/naive              5
        # /AllCells/DN                           0
        # /AllCells/DP                         973
        node_percentages = ncells_per_node_per_sample.div(ncells_per_sample, axis=1)
        #                         single_sample
        # /AllCells/CD4+/CD8-/Tcm         0.000000
        # /AllCells/CD4+/CD8-/Tem         0.000000
        # /AllCells/CD4+/CD8-/Temra       0.001001
        # /AllCells/CD4+/CD8-/naive       0.020020
        # /AllCells/CD4-/CD8+/Tcm         0.000000
        # /AllCells/CD4-/CD8+/Tem         0.000000
        # /AllCells/CD4-/CD8+/Temra       0.000000
        # /AllCells/CD4-/CD8+/naive       0.005005
        # /AllCells/DN                    0.000000
        # /AllCells/DP                    0.973974

        FlowSimulationTreeDirichlet(
            node_percentages=node_percentages,
            rootnode=celltree,
            include_features="dataset_melanoma_short",
        )

    def test_FlowSimulationTree_verbosity(self):
        celltree = nbtree.tree_complete_aligned_trunk()
        celltree.data = self.cellmat
        celltree.id_preds(celltree.predict())

        FlowSimulationTreeDirichlet(
            node_percentages=None,
            rootnode=celltree,
            # With data_cellgroup_col=None,
            # all cells are assumed to come from the same sample
            data_cellgroup_col=None,
            include_features="dataset_melanoma_short",
            verbose=True,
        )

    def test_FlowSimulationTree_sample_populations(self):
        celltree = nbtree.tree_complete_aligned_trunk()
        celltree.data = self.cellmat
        celltree.id_preds(celltree.predict())

        flowsim = FlowSimulationTreeDirichlet(
            node_percentages=None,
            rootnode=celltree,
            # With data_cellgroup_col=None,
            # all cells are assumed to come from the same sample
            data_cellgroup_col=None,
            include_features="dataset_melanoma_short",
            verbose=True,
        )
        assert sum(flowsim.sample_populations(n_cells=10)) == 10
        assert sum(flowsim.sample_populations()) == 10000
        # Sampling twice 10k cells should give different results
        assert not flowsim.sample_populations().equals(flowsim.sample_populations())

        # Sampling twice with the same seed must give identical results
        flowsim.set_seed(41237)
        tmp1 = flowsim.sample_populations()
        flowsim.set_seed(41237)
        tmp2 = flowsim.sample_populations()
        assert tmp1.equals(tmp2)

    def test_FST_sample_pop_params(self):
        flowsim = self.flowsim_trunk

        assert flowsim.sample(n_cells=10).shape == (10, 13)
        assert flowsim.sample(n_cells=100).shape == (100, 13)

        flowsim.set_seed(41237)
        a1 = flowsim.sample()

        flowsim.set_seed(41237)
        a2 = flowsim.sample(**flowsim.population_parameters)
        assert a1.equals(a2)

        customized_pop_params = copy.deepcopy(flowsim.population_parameters)
        flowsim.set_seed(41237)
        a3 = flowsim.sample(**customized_pop_params)
        assert a1.equals(a3)

        customized_pop_params["alpha"] *= 100
        flowsim.set_seed(41237)
        a4 = flowsim.sample(**customized_pop_params)
        # The underlying population parameters were actively changed,
        # so the samples should be different
        assert not a1.equals(a4)

    def test_FST_sample_pop_return_cell_numbers(self):
        flowsim = self.flowsim_trunk
        flowsim.set_seed(41237)
        a0_counts = flowsim.sample_populations(n_cells=10000)
        flowsim.set_seed(41237)
        a5, a5_counts = flowsim.sample(return_sampled_cell_numbers=True)
        flowsim.set_seed(41237)
        a6, a6_counts = flowsim.sample(
            return_sampled_cell_numbers=True, use_only_diagonal_covmat=False
        )

        assert a0_counts.equals(a5_counts)
        # The cells per population are either sampled with or without (only) diagonal
        # covariance matrix. That changes how the cells look - but not how many
        # cells per population.
        assert not a5.equals(a6)
        assert a5_counts.equals(a6_counts)

    def test_FST_sample_pop_covmat_diagonal(self):
        flowsim = self.flowsim_trunk
        flowsim.set_seed(41237)
        a5, a5_counts = flowsim.sample(return_sampled_cell_numbers=True)
        flowsim.set_seed(41237)
        a6, a6_counts = flowsim.sample(
            return_sampled_cell_numbers=True, use_only_diagonal_covmat=False
        )

        # The cells per population are either sampled with or without (only) diagonal
        # covariance matrix. That changes how the cells look - but not how many
        # cells per population.
        assert not a5.equals(a6)
        assert a5_counts.equals(a6_counts)
