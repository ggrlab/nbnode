import os
from unittest import TestCase

import numpy as np
import pandas as pd
import anytree

from nbnode_pyscaffold.simulation.FlowSimulationTree import FlowSimulationTreeDirichlet
from nbnode_pyscaffold.nbnode import NBNode
from nbnode_pyscaffold.nbnode_trees import (
    tree_simple,
    tree_complex,
    tree_simpleB,
)
from nbnode_pyscaffold.testutil.helpers import find_dirname_above_currentfile


TESTS_DIR = find_dirname_above_currentfile()


class TestFlowSimulation(TestCase):
    def test_tree_simple_predict(self):
        mytree = tree_simple()

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
        celltree = tree_complex()
        yternary_preds = celltree.predict(values=yternary)
        celltree.id_preds(yternary_preds)

        yternary["fake_activations"] = np.random.default_rng(12936).random(
            len(yternary)
        )
        celltree.data = yternary
        flow_dist = FlowSimulationTreeDirichlet(
            rootnode=celltree,
            include_features=["CD57", "PD1", "fake_activations"],
            node_percentages=None,
            data_cellgroup_col="CD8",  # JUST FOR TESTING!
        )

    def test_FlowSimulationTreeDirichlet(self):
        yternary = pd.read_csv(
            os.path.join(
                TESTS_DIR, "testdata", "flowcytometry", "gated_cells", "yternary.csv"
            )
        )
        celltree = tree_complex()
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

        flow_dist = FlowSimulationTreeDirichlet(
            rootnode=celltree,
            include_features=["CD57", "PD1", "fake_activations"],
            node_percentages=None,
            data_cellgroup_col="CD8",  # JUST FOR TESTING!
        )
        # test sampling a single sample
        flow_dist.set_seed(1230847)
        x = flow_dist.sample(n_cells=1)
        print(x.iloc[0].to_numpy())
        assert all(
            np.isclose(
                x.iloc[0].to_numpy(), np.array([0.9913707, -0.78346626, 0.07124489])
            )
        )
        x = flow_dist.sample(n_cells=100)
        assert not all(
            np.isclose(
                x.iloc[0].to_numpy(), np.array([0.9913707, -0.78346626, 0.07124489])
            )
        )
        flow_dist.set_seed(1230847)
        x = flow_dist.sample(n_cells=1)
        assert all(
            np.isclose(
                x.iloc[0].to_numpy(), np.array([0.9913707, -0.78346626, 0.07124489])
            )
        )
        assert x.shape[0] == 1
        assert x.shape[1] == 3
        flow_dist.set_seed(1230847)
        x = flow_dist.sample(n_cells=100)
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
        celltree = tree_complex()
        yternary_preds = celltree.predict(values=yternary)
        celltree.id_preds(yternary_preds)

        yternary["fake_activations"] = np.random.default_rng(12936).random(
            len(yternary)
        )
        celltree.data = yternary
        flow_dist = FlowSimulationTreeDirichlet(
            rootnode=celltree,
            include_features=["CD57", "PD1", "fake_activations"],
            node_percentages=None,
            data_cellgroup_col=None,  # Then all cells are assumed to come from the same sample
        )

        x = flow_dist.sample(n_cells=100)
        assert x.shape[0] == 100
        assert x.shape[1] == 3

    def test_rng_numpy(self):
        a = np.random.default_rng(120983)
        newnumber = a.random(1)
        assert np.isclose(newnumber, 0.71607698)
        assert not np.isclose(newnumber, a.random(1))
        a = np.random.default_rng(120983)
        assert np.isclose(0.71607698, a.random(1))

    def test_FlowSimulationTreeDirichlet(self):
        yternary = pd.read_csv(
            os.path.join(
                TESTS_DIR, "testdata", "flowcytometry", "gated_cells", "yternary.csv"
            )
        )
        celltree = tree_complex()
        yternary_preds = celltree.predict(values=yternary)
        celltree.id_preds(yternary_preds)

        yternary["fake_activations"] = np.random.default_rng(12936).random(
            len(yternary)
        )
        celltree.data = yternary

        leaf_nodes = [x for x in anytree.PreOrderIter(celltree) if x.is_leaf]

        flow_dist = FlowSimulationTreeDirichlet(
            rootnode=celltree,
            include_features=["CD57", "PD1", "fake_activations"],
            node_percentages=None,
            data_cellgroup_col=None,  # Then all cells are assumed to come from the same sample
        )

        flow_dist.set_seed(10289)
        z = flow_dist.sample(
            n_cells=100,
            # mean=flow_dist.population_distribution_parameters["mean"] * 2,
            # cov=np.diag(np.diag(flow_dist.population_distribution_parameters["cov"])),
            # population_names=flow_dist.population_distribution_parameters["mean"].index,
        )

    def test_subset_by_nodename(self):
        mytree = tree_simple()
        for node_x in anytree.PreOrderIter(mytree):
            node_x: NBNode
            a = mytree[node_x.get_name_full()]
            print(a)
        assert mytree["nonexisting_node_name"] is None
        with self.assertRaises(ValueError):
            mytree["illegal_two_elements", "illegal_two_elements"]

    def test_FlowSimulationTreeDirichlet_return_sampled_cell_numbers(self):
        mytree = tree_simpleB()
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

        flow_dist = FlowSimulationTreeDirichlet(
            rootnode=mytree,
            include_features=["f1", "f2"],
            node_percentages=None,
            data_cellgroup_col=None,  # Then all cells are assumed to come from the same sample
        )
        flow_dist.set_seed(1234)
        leaf_sample = flow_dist.sample(10, return_sampled_cell_numbers=True)
        leaf_sample_popN = leaf_sample[1]
        assert len(leaf_sample_popN) == 2

    def test_FlowSimulationTree_dirichlet(self):
        yternary = pd.read_csv(
            os.path.join(
                TESTS_DIR, "testdata", "flowcytometry", "gated_cells", "yternary.csv"
            )
        )
        celltree = tree_complex()
        yternary_preds = celltree.predict(values=yternary)
        celltree.id_preds(yternary_preds)

        yternary["fake_activations"] = np.random.default_rng(12936).random(
            len(yternary)
        )
        celltree.data = yternary

        flow_dist = FlowSimulationTreeDirichlet(
            include_features=["CD57", "PD1", "fake_activations"],
            node_percentages=None,
            rootnode=celltree,
            data_cellgroup_col=None,  # Then all cells are assumed to come from the same sample
        )

        flow_dist.set_seed(10289)
        z = flow_dist.sample(n_cells=100)

    def test_FlowSimulationTree_report_mean_precision(self):
        yternary = pd.read_csv(
            os.path.join(
                TESTS_DIR, "testdata", "flowcytometry", "gated_cells", "yternary.csv"
            )
        )
        celltree = tree_complex()
        yternary_preds = celltree.predict(values=yternary)
        celltree.id_preds(yternary_preds)

        yternary["fake_activations"] = np.random.default_rng(12936).random(
            len(yternary)
        )
        celltree.data = yternary

        flow_dist = FlowSimulationTreeDirichlet(
            include_features=["CD57", "PD1", "fake_activations"],
            node_percentages=None,
            rootnode=celltree,
            data_cellgroup_col=None,  # Then all cells are assumed to come from the same sample
        )
        print(flow_dist.precision)
        print(flow_dist.mean_leafs)
        assert flow_dist.pop_alpha("/AllCells") == flow_dist.precision
        assert (
            flow_dist.pop_alpha("/AllCells/CD45+")
            == flow_dist.precision
            - flow_dist.population_parameters["alpha"]["/AllCells/not CD45"]
        )
        assert flow_dist.pop_mean("/AllCells") == 1
        assert (
            flow_dist.pop_mean("/AllCells/CD45+")
            == (
                flow_dist.precision
                - flow_dist.population_parameters["alpha"]["/AllCells/not CD45"]
            )
            / flow_dist.precision
        )

    def test_FlowSimulationTree_report_all_alpha(self):
        yternary = pd.read_csv(
            os.path.join(
                TESTS_DIR, "testdata", "flowcytometry", "gated_cells", "yternary.csv"
            )
        )
        celltree = tree_complex()
        yternary_preds = celltree.predict(values=yternary)
        celltree.id_preds(yternary_preds)

        yternary["fake_activations"] = np.random.default_rng(12936).random(
            len(yternary)
        )
        celltree.data = yternary

        flow_dist = FlowSimulationTreeDirichlet(
            include_features=["CD57", "PD1", "fake_activations"],
            node_percentages=None,
            rootnode=celltree,
            data_cellgroup_col=None,  # Then all cells are assumed to come from the same sample
        )
        print(flow_dist.precision)
        print(flow_dist.mean_leafs)
        print(flow_dist.alpha_all)

    def test_FlowSimulationTree_new_mean(self):
        yternary = pd.read_csv(
            os.path.join(
                TESTS_DIR, "testdata", "flowcytometry", "gated_cells", "yternary.csv"
            )
        )
        celltree = tree_complex()
        yternary_preds = celltree.predict(values=yternary)
        celltree.id_preds(yternary_preds)

        yternary["fake_activations"] = np.random.default_rng(12936).random(
            len(yternary)
        )
        celltree.data = yternary

        flow_dist = FlowSimulationTreeDirichlet(
            include_features=["CD57", "PD1", "fake_activations"],
            node_percentages=None,
            rootnode=celltree,
            data_cellgroup_col=None,  # Then all cells are assumed to come from the same sample
        )
        # print(flow_dist.precision)
        assert len(flow_dist.pop_leafnode_names("/AllCells")) == len(
            [x for x in anytree.PreOrderIter(celltree) if x.is_leaf]
        )
        assert len(flow_dist.pop_leafnode_names("/AllCells/not CD45")) == 1
        assert (
            len(flow_dist.pop_leafnode_names("/AllCells/CD45+"))
            == len([x for x in anytree.PreOrderIter(celltree) if x.is_leaf]) - 1
        )
        assert len(flow_dist.pop_leafnode_names("/AllCells/CD45+/CD3+/CD4+/CD8-")) == 1

        with self.assertRaises(ValueError):
            flow_dist.new_pop_mean("/AllCells/not CD45", percentage=1.1)
        with self.assertRaises(ValueError):
            flow_dist.new_pop_mean("/AllCells/not CD45", percentage=-0.1)

        flow_dist.new_pop_mean("/AllCells/not CD45", percentage=0.1)
        assert flow_dist.pop_mean("/AllCells/not CD45") == 0.1

        flow_dist.new_pop_mean("/AllCells/CD45+/CD3+", percentage=0.1)
        assert flow_dist.pop_mean("/AllCells/CD45+/CD3+") == 0.1

    def test_FlowSimulationTree_report_parameters(self):
        yternary = pd.read_csv(
            os.path.join(
                TESTS_DIR, "testdata", "flowcytometry", "gated_cells", "yternary.csv"
            )
        )
        celltree = tree_complex()
        yternary_preds = celltree.predict(values=yternary)
        celltree.id_preds(yternary_preds)

        yternary["fake_activations"] = np.random.default_rng(12936).random(
            len(yternary)
        )
        celltree.data = yternary

        flow_dist = FlowSimulationTreeDirichlet(
            include_features=["CD57", "PD1", "fake_activations"],
            node_percentages=None,
            rootnode=celltree,
            data_cellgroup_col=None,  # Then all cells are assumed to come from the same sample
        )
        for x in anytree.PreOrderIter(celltree):
            print(
                x.get_name_full(),
                flow_dist.pop_alpha(x.get_name_full()),
                flow_dist.pop_mean(x.get_name_full()),
            )
    def test_FlowSimulationTree_reset_populations(self):
        yternary = pd.read_csv(
            os.path.join(
                TESTS_DIR, "testdata", "flowcytometry", "gated_cells", "yternary.csv"
            )
        )
        celltree = tree_complex()
        yternary_preds = celltree.predict(values=yternary)
        celltree.id_preds(yternary_preds)

        yternary["fake_activations"] = np.random.default_rng(12936).random(
            len(yternary)
        )
        celltree.data = yternary

        flow_dist = FlowSimulationTreeDirichlet(
            include_features=["CD57", "PD1", "fake_activations"],
            node_percentages=None,
            rootnode=celltree,
            data_cellgroup_col=None,  # Then all cells are assumed to come from the same sample
        )
        assert flow_dist.pop_mean("/AllCells/not CD45") == 0.28343526674499486
        flow_dist.reset_populations()
        assert flow_dist.pop_mean("/AllCells/not CD45") == 0.28343526674499486

        flow_dist.new_pop_mean("/AllCells/not CD45", percentage=0.1)
        assert flow_dist.pop_mean("/AllCells/not CD45") == 0.1
        
        flow_dist.reset_populations()
        assert flow_dist.pop_mean("/AllCells/not CD45") == 0.28343526674499486
