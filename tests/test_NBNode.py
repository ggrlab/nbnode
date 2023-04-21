import os
from unittest import TestCase

import anytree

# from anytree.exporter import a_exp.DotExporter
import anytree.exporter as a_exp

import nbnode_pyscaffold.nbnode_trees as nbtree
from nbnode_pyscaffold.nbnode import NBNode
from nbnode_pyscaffold.testutil.helpers import find_dirname_above_currentfile

TESTS_DIR = find_dirname_above_currentfile()


class TestNBNode(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # https://docs.python.org/3/library/unittest.html#unittest.TestCase.setUpClass
        # setUpClass is only called once for the whole class in contrast to setUp which
        # is called before every test.

        os.makedirs("tests_output", exist_ok=True)

    def test_create_tree_simple(self):
        mytree = nbtree.tree_simple()
        for pre, _, node in anytree.RenderTree(mytree):
            print(
                "%s%s (%s %s)"
                % (pre, node.name, node.decision_name, node.decision_value or 0)
            )
        a_exp.DotExporter(
            mytree,
            nodenamefunc=lambda node: node.name,
            edgeattrfunc=lambda parent, child: 'label="%s: %s"'
            % (child.decision_name, str(child.decision_value))
            # edgeattrfunc=lambda parent, child: "style=bold,label=%s"
            # % (child.decision_name or "NA"),
        ).to_picture(
            "tests_output/tree_simple.pdf"
        )  # doctest: +SKIP
        a_exp.DotExporter(
            mytree,
            nodenamefunc=lambda node: node.name,
            edgeattrfunc=lambda parent, child: 'label="%s: %s"'
            % (child.decision_name, str(child.decision_value))
            # edgeattrfunc=lambda parent, child: "style=bold,label=%s"
            # % (child.decision_name or "NA"),
        ).to_dotfile(
            "tests_output/tree_simple.txt"
        )  # doctest: +SKIP

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

    def test_tree_simple_predict_str(self):
        mytree = nbtree.tree_simple()
        mytree.pretty_print()

        single_prediction = mytree.prediction_str("/a")
        assert single_prediction.name == "a"
        single_prediction = mytree.prediction_str("/a/a0")
        assert single_prediction.name == "a0"
        single_prediction = mytree.prediction_str("/a/a1")
        assert single_prediction.name == "a1"
        single_prediction = mytree.prediction_str("/a/a1/a1a")
        assert single_prediction.name == "a1a"
        single_prediction = mytree.prediction_str("a/a1/a1a")
        assert single_prediction.name == "a1a"

        with self.assertRaises(ValueError):
            # Try to predict a hierarchy without the root node (must fail)
            single_prediction = mytree.prediction_str("a1/a1a")

        # If the same (failed) prediction done is on the "correct" part of the tree,
        # it works!
        # ("predict" only with the subpart starting from a1)
        single_prediction = mytree.children[1].prediction_str("a1/a1a")
        assert single_prediction.name == "a1a"

    def test_unfitting_data(self):
        mytree = NBNode("a")
        # a0 =
        NBNode("a0", parent=mytree, decision_value=-1, decision_name="m1")
        # a1 =
        NBNode("a1", parent=mytree, decision_value=1, decision_name="m1")

        single_prediction = mytree.predict(values=[-1], names=["m1"])
        assert single_prediction.name == "a0"
        single_prediction = mytree.predict(values=[1], names=["m1"])
        assert single_prediction.name == "a1"
        # Now try to predict an unknown value, raises error because
        # no fitting endnote can be found.
        with self.assertRaises(ValueError):
            single_prediction = mytree.predict(values=[-999], names=["m1"])
        single_prediction = mytree.predict(
            values=[-999], names=["m1"], allow_unfitting_data=True
        )
        # print(single_prediction)
        assert single_prediction == []

        mytree = nbtree.tree_simple()
        # The following goes to
        #   a -> a1 -(NOT because m2 != 'test')> a1a (endnode)
        #   a -> a2 (endnode)
        with self.assertRaises(ValueError):
            # This prediction here fails because:
            # 1.1 Check if m1=-1 (no)
            # 2.1 Check if m1=1 (yes)
            # 2.2 Check if m2='test' (no), no endnode!
            #   raise exception because in this path no proper endnode was able to be
            #   found with the given values
            # 3.1 Check if m3='another' (yes) -> return this node
            single_prediction = mytree.predict(
                values={"m1": 1, "m2": -1, "m3": "another"}
            )

        single_prediction = mytree.predict(
            values={"m1": 1, "m2": -1, "m3": "another"}, allow_unfitting_data=True
        )
        # print(single_prediction)
        # print(single_prediction.name)

        assert single_prediction.name == "a2"

    def test_repr(self):
        # What if we want only part of the predictions, not the end-nodes?
        mytree = nbtree.tree_simple()
        assert (
            mytree.__repr__()
            == "NBNode('/a', counter=0, decision_name=None, decision_value=None)"
        )

    def test_part_prediction(self):
        # What if we want only part of the predictions, not the end-nodes?
        mytree = nbtree.tree_simple()
        # Goes to
        #   a -(m1=-1)-> a0 (endnode)
        #   a -(m1=1)-> a1 -(m2='test')-> a1a (endnode)
        #   a -(m3='another')-> a2 (endnode)

        # Usually that raises
        with self.assertRaises(ValueError):
            single_prediction = mytree.predict(
                values=[1, 0, 0], names=["m1", "m2", "m3"]
            )

        # Finding the endnode and going backwards
        single_prediction = mytree.predict(
            values=[1, "test", 0], names=["m1", "m2", "m3"]
        )
        # reverse_path_nodes =
        [x for x in single_prediction.iter_path_reverse()]

        # print([x.name for x in reverse_path_nodes])

        # Finding parts
        single_prediction = mytree.predict(
            values=[1, 0, "another22"],
            names=["m1", "m2", "m3"],
            allow_part_predictions=True,
        )
        print(single_prediction)
        assert isinstance(single_prediction, NBNode)

        # Finding matching parts AND a matching endnode
        single_prediction = mytree.predict(
            values=[1, 0, "another"],
            names=["m1", "m2", "m3"],
            allow_part_predictions=True,
        )
        assert isinstance(single_prediction, list)
        assert len(single_prediction) == 2

    def test_multiple_endnodes(self):
        mytree = nbtree.tree_simple()
        # Goes to
        #   a -> a1 -> a1a (endnote)
        #   a -> a2 (endnote)
        single_prediction = mytree.predict(
            values={"m1": 1, "m2": "test", "m3": "another"}, allow_unfitting_data=False
        )
        assert [x.name for x in single_prediction] == ["a1a", "a2"]

    def test_missing_values(self):
        mytree = nbtree.tree_simple()

        with self.assertRaises(ValueError):
            # m1==-1 is a valid branch. But the tree would also like to
            # check if 'm3'=='another'.
            # But the value for 'm3' is not defined, therefore an exception
            # must be raised.

            # single_prediction =
            mytree.predict(values=[-1], names=["m1"])

    def test_non_binary_branch(self):
        celltree = nbtree.tree_complex()
        tmp = celltree.predict({"CD45": 1, "CD3": 1, "CD4": -1, "CD8": -1})
        print(tmp.name)

    def test_part_insertion(self):
        celltree = nbtree.tree_complex()
        temra_part = [
            NBNode("naive", decision_name=["CCR7", "CD45RA"], decision_value=[1, 1]),
            NBNode("Tcm", decision_name=["CCR7", "CD45RA"], decision_value=[1, -1]),
            NBNode("Temra", decision_name=["CCR7", "CD45RA"], decision_value=[-1, 1]),
            NBNode("Tem", decision_name=["CCR7", "CD45RA"], decision_value=[-1, -1]),
        ]
        temra_part2 = [
            NBNode("naive", decision_name=["CCR7", "CD45RA"], decision_value=[1, 1]),
            NBNode("Tcm", decision_name=["CCR7", "CD45RA"], decision_value=[1, -1]),
            NBNode("Temra", decision_name=["CCR7", "CD45RA"], decision_value=[-1, 1]),
            NBNode("Tem", decision_name=["CCR7", "CD45RA"], decision_value=[-1, -1]),
        ]
        cd4NEG_cd8POS_node = celltree.predict(
            {"CD45": 1, "CD3": 1, "CD4": -1, "CD8": 1}
        )
        cd4POS_cd8NEG_node = celltree.predict(
            {"CD45": 1, "CD3": 1, "CD4": 1, "CD8": -1}
        )

        cd4POS_cd8NEG_node.insert_nodes(temra_part)

        with self.assertWarns(Warning):
            # Here a warning must come for list reusage
            cd4NEG_cd8POS_node.insert_nodes(temra_part)

        import warnings

        # check that NO warnings occur
        with warnings.catch_warnings(record=True) as w:
            # the following is fine
            cd4NEG_cd8POS_node.insert_nodes(temra_part2)
            # this is also fine
            temra_part = [
                NBNode(
                    "naive", decision_name=["CCR7", "CD45RA"], decision_value=[1, 1]
                ),
                NBNode("Tcm", decision_name=["CCR7", "CD45RA"], decision_value=[1, -1]),
                NBNode(
                    "Temra", decision_name=["CCR7", "CD45RA"], decision_value=[-1, 1]
                ),
                NBNode(
                    "Tem", decision_name=["CCR7", "CD45RA"], decision_value=[-1, -1]
                ),
            ]
            cd4NEG_cd8POS_node.insert_nodes(temra_part, copy_list=True)
            cd4NEG_cd8POS_node.insert_nodes(temra_part, copy_list=True)
            if len(w) != 0:
                raise AssertionError("Inserting nodes issued a warning")
        warnings.warn(
            "It can still be that you have the same decisions multiple times, "
            "I did not test so far what happens then. (2021-02-06)"
        )

    def test_part_insertion_complex(self):
        celltree = nbtree.tree_complete_cell()
        print(anytree.RenderTree(celltree))

        a_exp.DotExporter(
            celltree,
            nodenamefunc=lambda node: node.name,
            edgeattrfunc=lambda parent, child: 'label="%s: %s"'
            % (child.decision_name, child.decision_value),
        ).to_picture(
            "tests_output/celltree.pdf"
        )  # doctest: +SKIP

        a_exp.DotExporter(
            celltree,
            nodeattrfunc=lambda node: 'label="{}"'.format(node.name),
        ).to_picture(
            "tests_output/celltree_no_unique_labels.pdf"
        )  # doctest: +SKIP
        a_exp.UniqueDotExporter(
            celltree,
            nodeattrfunc=lambda node: 'label="{}"'.format(node.name),
        ).to_picture(
            "tests_output/celltree_unique_dotexporter.pdf"
        )  # doctest: +SKIP

    def test_plot_NBNode_tree_coloring(self):
        import pydotplus

        from nbnode_pyscaffold.plot.utils import plot_save_unified

        simpletree = nbtree.tree_simple()
        dot_data = a_exp.UniqueDotExporter(
            simpletree,
            options=['node [shape=box, style="filled", color="black"];'],
            nodeattrfunc=lambda node: 'label="{}", fillcolor="white"'.format(node.name),
        )
        dotdata_str = "\n".join([x for x in dot_data])
        # print(dotdata_str)
        graph: pydotplus.Dot = pydotplus.graph_from_dot_data(dotdata_str)
        nodes = graph.get_node_list()

        plot_save_unified(
            any_plot=graph, file="tests_output/pydotplus_graph_nocolor.pdf"
        )
        for node in nodes:
            if node.get_name() not in ("node", "edge", "plottitle"):
                node.set_fillcolor("#ff5a00")
        # print(graph.to_string())
        plot_save_unified(any_plot=graph, file="tests_output/pydotplus_graph_color.pdf")

    def test_dotexport(self):
        simpletree = nbtree.tree_simple()
        exported_dotstr = simpletree.export_dot()
        dot_data = a_exp.UniqueDotExporter(
            simpletree,
            options=['node [shape=box, style="filled", color="black"];'],
            nodeattrfunc=lambda node: 'label="{}", fillcolor="white"'.format(node.name),
        )
        dotdata_str = "\n".join([x for x in dot_data])
        assert exported_dotstr == dotdata_str

    def test_apply_NBNode_tree_to_dataframe(self):
        import pandas as pd

        yternary = pd.read_csv(
            os.path.join(
                TESTS_DIR, "testdata", "flowcytometry", "gated_cells", "yternary.csv"
            )
        )
        print(yternary)

        celltree = nbtree.tree_complete_cell()
        print(
            celltree.predict(values=yternary.iloc[0, :], names=list(yternary.columns))
        )
        print(celltree.predict(values=yternary))
        # tmp =
        celltree.predict(values=yternary)

    def test_apply_NBNode_tree_to_nparray(self):
        import pandas as pd

        yternary = pd.read_csv(
            os.path.join(
                TESTS_DIR, "testdata", "flowcytometry", "gated_cells", "yternary.csv"
            )
        )
        yternary_colnames = list(yternary.columns)
        yternary_np = yternary.to_numpy()
        celltree = nbtree.tree_complete_cell()
        print(celltree.predict(values=yternary_np[0, :], names=yternary_colnames))

        # check if numpy prediction is the same as pandas prediction
        assert list(
            celltree.predict(values=yternary_np, names=yternary_colnames)
        ) == list(celltree.predict(values=yternary))

    def test_count_nodes(self):
        import pandas as pd

        yternary = pd.read_csv(
            os.path.join(
                TESTS_DIR, "testdata", "flowcytometry", "gated_cells", "yternary.csv"
            )
        )
        celltree = nbtree.tree_complete_cell()
        yternary_preds = celltree.predict(values=yternary)
        celltree.count(yternary_preds)
        assert celltree.counter == 999
        # print(tmp)
        celltree.count(yternary_preds, reset_counts=False)
        assert celltree.counter == 1998
        celltree.count(yternary_preds, reset_counts=True)
        print(celltree.counter)
        assert celltree.counter == 999

    def test_count_nodes_store_ids(self):
        import pandas as pd

        yternary = pd.read_csv(
            os.path.join(
                TESTS_DIR, "testdata", "flowcytometry", "gated_cells", "yternary.csv"
            )
        )
        celltree = nbtree.tree_complete_cell()
        yternary_preds = celltree.predict(values=yternary)
        celltree.count(yternary_preds, reset_counts=True)
        # print(celltree.counter)
        assert celltree.counter == 999
        celltree.id_preds(yternary_preds)
        assert celltree.ids == [*range(999)]
        # print(celltree.children[0].counter)

    def test_reset_ids(self):
        import pandas as pd

        yternary = pd.read_csv(
            os.path.join(
                TESTS_DIR, "testdata", "flowcytometry", "gated_cells", "yternary.csv"
            )
        )
        celltree = nbtree.tree_complete_cell()
        assert len(celltree.ids) == 0
        assert len(celltree.children[0].ids) == 0
        yternary_preds = celltree.predict(values=yternary)
        assert len(celltree.ids) == 0
        assert len(celltree.children[0].ids) == 0
        celltree.count(yternary_preds, reset_counts=True)
        assert len(celltree.ids) == 0
        assert len(celltree.children[0].ids) == 0
        assert celltree.counter == 999
        celltree.id_preds(yternary_preds)
        assert len(celltree.ids) == 999
        assert len(celltree.children[0].ids) == 301
        celltree.reset_ids()
        assert len(celltree.ids) == 0
        assert len(celltree.children[0].ids) == 0

    def test_counting_on_ids(self):
        import pandas as pd

        yternary = pd.read_csv(
            os.path.join(
                TESTS_DIR, "testdata", "flowcytometry", "gated_cells", "yternary.csv"
            )
        )
        celltree = nbtree.tree_complete_cell()
        yternary_preds = celltree.predict(values=yternary)
        import copy

        celltree.count(yternary_preds, reset_counts=True)
        counted_tree_counter = copy.deepcopy(celltree)

        celltree.id_preds(yternary_preds)
        celltree.count(yternary_preds, reset_counts=True, use_ids=True)
        counted_tree_ids = copy.deepcopy(celltree)
        assert [
            x.counter for x in anytree.iterators.PreOrderIter(counted_tree_counter)
        ] == [x.counter for x in anytree.iterators.PreOrderIter(counted_tree_ids)]

        # print([x.counter for x in anytree.iterators.PreOrderIter(counted_tree_ids)])

    def test_id_based_data(self):
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
        assert len(celltree.children[0].data) == 301
        assert len(celltree.data.columns) == 11
        assert len(celltree.children[0].data.columns) == 11

    def test_id_based_data_child(self):
        # Even when setting the data property for a child, the data is always stored
        # in the ROOT node and accessed from the children via the predicted ids.
        import pandas as pd

        yternary = pd.read_csv(
            os.path.join(
                TESTS_DIR, "testdata", "flowcytometry", "gated_cells", "yternary.csv"
            )
        )
        celltree = nbtree.tree_complete_cell()
        yternary_preds = celltree.predict(values=yternary)
        celltree.id_preds(yternary_preds)
        celltree.children[0].data = yternary
        assert len(celltree.data) == 999
        assert len(celltree.children[0].data) == 301
        assert len(celltree.data.columns) == 11
        assert len(celltree.children[0].data.columns) == 11

    def test_id_based_data_missing_ids(self):
        import pandas as pd

        yternary = pd.read_csv(
            os.path.join(
                TESTS_DIR, "testdata", "flowcytometry", "gated_cells", "yternary.csv"
            )
        )
        celltree = nbtree.tree_complete_cell()
        celltree.data = yternary
        # The len is 0 because I cannot select any samples for any node
        # (because the ids are all empty lists [] per initialization)
        assert len(celltree.data) == 0

    def test_data_summary(self):
        import pandas as pd

        yternary = pd.read_csv(
            os.path.join(
                TESTS_DIR, "testdata", "flowcytometry", "gated_cells", "yternary.csv"
            )
        )
        celltree = nbtree.tree_complete_cell()
        yternary_preds = celltree.predict(values=yternary)
        celltree.id_preds(yternary_preds)
        celltree.data = yternary

        def testfun(df: pd.DataFrame):
            return list(df.apply(sum, axis=0))

        # no_attribute_result =
        celltree.apply(fun=testfun)
        with self.assertRaises(AttributeError):
            # for key, value in no_attribute_result.items():
            #     print(value)
            print(celltree.new_test_attribute)

        # with setting the attribute
        celltree.apply(fun=testfun, result_attribute_name="new_test_attribute")
        print(celltree.new_test_attribute)

    def test_data_summary_plotting(self):
        import numpy as np
        import pandas as pd

        yternary = pd.read_csv(
            os.path.join(
                TESTS_DIR, "testdata", "flowcytometry", "gated_cells", "yternary.csv"
            )
        )
        celltree = nbtree.tree_complete_cell()
        yternary_preds = celltree.predict(values=yternary)
        celltree.id_preds(yternary_preds)
        yternary["fake_activations"] = np.random.random(len(yternary))
        celltree.data = yternary

        def mean_activation(df: pd.DataFrame):
            # from the following I saw that NaN only happen for df of length 0
            # print(len(df), df['fake_activations'].mean())
            return df["fake_activations"].mean()

        no_attribute_result = celltree.apply(fun=mean_activation)
        for key, value in no_attribute_result.items():
            print(value)

        # with setting the attribute
        celltree.apply(fun=mean_activation, result_attribute_name="new_test_attribute")

    def test_graph_from_dot(self):
        mytree = nbtree.tree_simple()
        # dotexported =
        mytree.export_dot()
        graph = mytree.graph_from_dot(mytree)
        from nbnode_pyscaffold.plot.utils import plot_save_unified

        plot_save_unified(
            any_plot=graph, file="tests_output/graph_from_dot_colored.pdf"
        )

    def test_graph_from_dot_summary_color(self):
        import numpy as np
        import pandas as pd

        yternary = pd.read_csv(
            os.path.join(
                TESTS_DIR, "testdata", "flowcytometry", "gated_cells", "yternary.csv"
            )
        )
        celltree = nbtree.tree_complete_cell()
        yternary_preds = celltree.predict(values=yternary)
        celltree.id_preds(yternary_preds)
        yternary["fake_activations"] = np.random.random(len(yternary))
        celltree.data = yternary

        def mean_activation(df: pd.DataFrame):
            # from the following I saw that NaN only happen for df of length 0
            # print(len(df), df['fake_activations'].mean())
            return df["fake_activations"].mean()

        # with setting the attribute
        celltree.apply(
            fun=mean_activation, result_attribute_name="fake_activation_mean"
        )

        graph = celltree.graph_from_dot(
            celltree, fillcolor_node_attribute="fake_activation_mean"
        )
        from nbnode_pyscaffold.plot.utils import plot_save_unified

        plot_save_unified(any_plot=graph, file="tests_output/fake_activation_mean.pdf")

    def test_graph_apply_fun_args(self):
        import numpy as np
        import pandas as pd

        yternary = pd.read_csv(
            os.path.join(
                TESTS_DIR, "testdata", "flowcytometry", "gated_cells", "yternary.csv"
            )
        )
        celltree = nbtree.tree_complete_cell()
        yternary_preds = celltree.predict(values=yternary)
        celltree.id_preds(yternary_preds)
        yternary["fake_activations"] = np.random.random(len(yternary))
        celltree.data = yternary

        def mean_activation(df: pd.DataFrame, test_id, test_id_2):
            # from the following I saw that NaN only happen for df of length 0
            # print(len(df), df['fake_activations'].mean())
            return df["fake_activations"].mean()

        celltree.apply(
            fun=mean_activation,
            result_attribute_name="new_test_attribute",
            test_id=1,
            test_id_2=2,
        )

    def test_node_text_attributes(self):
        import numpy as np
        import pandas as pd

        yternary = pd.read_csv(
            os.path.join(
                TESTS_DIR, "testdata", "flowcytometry", "gated_cells", "yternary.csv"
            )
        )
        celltree = nbtree.tree_complete_cell()
        yternary_preds = celltree.predict(values=yternary)
        celltree.id_preds(yternary_preds)
        yternary["fake_activations"] = np.random.random(len(yternary))
        celltree.data = yternary

        def mean_activation(df: pd.DataFrame, test_id, test_id_2):
            # from the following I saw that NaN only happen for df of length 0
            # print(len(df), df['fake_activations'].mean())
            return df["fake_activations"].mean()

        celltree.apply(
            fun=mean_activation,
            result_attribute_name="mean_act",
            test_id=1,
            test_id_2=2,
        )
        graph = celltree.graph_from_dot(celltree, fillcolor_node_attribute="mean_act")
        from nbnode_pyscaffold.plot.utils import plot_save_unified

        plot_save_unified(
            any_plot=graph, file="tests_output/graph_text_attributes_default.pdf"
        )

        graph = celltree.graph_from_dot(
            celltree,
            fillcolor_node_attribute="mean_act",
            node_text_attributes=["name", "mean_act"],
        )
        plot_save_unified(
            any_plot=graph, file="tests_output/graph_text_attributes_act.pdf"
        )
        graph = celltree.graph_from_dot(
            celltree,
            fillcolor_node_attribute="mean_act",
            node_text_attributes={"name": "{}", "mean_act": "{:.2f}"},
        )
        plot_save_unified(
            any_plot=graph, file="tests_output/graph_text_attributes_act_fmtstring.pdf"
        )

    def test_celltree_princple_explained(self):
        import numpy as np
        import pandas as pd

        yternary = pd.read_csv(
            os.path.join(
                TESTS_DIR, "testdata", "flowcytometry", "gated_cells", "yternary.csv"
            )
        )
        yternary["fake_activations"] = np.random.random(len(yternary)) - 0.5

        celltree = nbtree.tree_complete_cell()
        yternary_preds = celltree.predict(values=yternary)
        celltree.id_preds(yternary_preds)
        celltree.data = yternary

        def mean_activation(df: pd.DataFrame):
            # from the following I saw that NaN only happen for df of length 0
            # print(len(df), df['fake_activations'].mean())
            return df["fake_activations"].mean()

        celltree.apply(fun=mean_activation, result_attribute_name="mean_act")

        graph = celltree.graph_from_dot(
            celltree,
            fillcolor_node_attribute="mean_act",
            node_text_attributes={"name": "{}", "mean_act": "{:.2f}"},
        )
        dot_full_with_edgelabels = celltree.export_dot(
            unique_dot_exporter_kwargs={
                "options": ['node [shape=box, style="filled", color="black"];'],
                "nodeattrfunc": lambda node: 'label="{}", fillcolor="white"'.format(
                    node.name
                ),
                "edgeattrfunc": lambda parent, child: 'label="%s: %s"'
                % (child.decision_name, str(child.decision_value)),
            }
        )
        graph_full_with_edgelabels = celltree.graph_from_dot(
            tree=celltree,
            exported_dot_graph=dot_full_with_edgelabels,
            fillcolor_node_attribute="mean_act",
            node_text_attributes={"name": "{}", "mean_act": "{:.2f}"},
        )

        celltree_stump = nbtree.tree_complex()
        yternary_preds = celltree_stump.predict(values=yternary)
        celltree_stump.id_preds(yternary_preds)
        celltree_stump.data = yternary

        def mean_activation(df: pd.DataFrame):
            # from the following I saw that NaN only happen for df of length 0
            # print(len(df), df['fake_activations'].mean())
            return df["fake_activations"].mean()

        celltree_stump.apply(fun=mean_activation, result_attribute_name="mean_act")
        dotstump = celltree_stump.export_dot(
            unique_dot_exporter_kwargs={
                "options": ['node [shape=box, style="filled", color="black"];'],
                "nodeattrfunc": lambda node: 'label="{}", fillcolor="white"'.format(
                    node.name
                ),
                "edgeattrfunc": lambda parent, child: 'label="%s: %s"'
                % (child.decision_name, str(child.decision_value)),
            }
        )
        graph_stump = celltree_stump.graph_from_dot(
            tree=celltree_stump,
            exported_dot_graph=dotstump,
            fillcolor_node_attribute="mean_act",
            node_text_attributes={"name": "{}", "mean_act": "{:.2f}"},
        )
        from nbnode_pyscaffold.plot.utils import plot_save_unified

        plot_save_unified(any_plot=graph_stump, file="tests_output/graph_stump.pdf")
        plot_save_unified(
            any_plot=graph_full_with_edgelabels, file="tests_output/graph_full.pdf"
        )
        plot_save_unified(any_plot=graph, file="tests_output/graph_celltree.pdf")

    def test_pretty_print(self):
        mytree = nbtree.tree_simple()
        mytree.pretty_print()
        mytree.pretty_print(print_attributes=["counter"])
        mytree.pretty_print(print_attributes=["counter", "counter"])
        mytree.pretty_print(print_attributes=None)

    def test_pretty_print_rounding(self):
        mytree = nbtree.tree_simple()
        mytree.counter = 15.41212
        mytree.pretty_print()
        mytree.pretty_print(print_attributes=["counter"])
        mytree.pretty_print(print_attributes=["counter", "counter"])
        mytree.pretty_print(round_ndigits=2)

    def test_len(self):
        mytree = NBNode("a")
        assert len(mytree) == 1
        a0 = NBNode("a0", parent=mytree, decision_value=-1, decision_name="m1")
        assert len(a0) == 1
        assert len(mytree) == 2
        a1 = NBNode("a1", parent=mytree, decision_value=1, decision_name="m1")
        assert len(mytree) == 3
        # a2 =
        NBNode("a2", parent=mytree, decision_value="another", decision_name="m3")
        assert len(mytree) == 4
        assert len(a1) == 1
        # a1a =
        NBNode("a1a", parent=a1, decision_value="test", decision_name="m2")
        assert len(a1) == 2
        assert len(mytree) == 5

    def test_both_iterator(self):
        mytree = nbtree.tree_simple()
        # for x in mytree.both_iterator(mytree):
        #     print(x)
        equal_lens = [x for x in mytree.both_iterator(mytree)]
        assert len(equal_lens) == len(mytree)

        mytree2 = nbtree.tree_complex()
        too_short_first = [x for x in mytree.both_iterator(mytree2)]
        assert len(too_short_first) == len(mytree)

        too_short_second = [x for x in mytree2.both_iterator(mytree)]
        assert len(too_short_second) == len(mytree)

    def test_eq_structure(self):
        mytree = nbtree.tree_simple()
        # mytree.pretty_print()

        mytree2 = nbtree.tree_complex()
        assert mytree.eq_structure(mytree)
        assert not mytree.eq_structure(mytree2)
        assert mytree2.eq_structure(mytree2)
        assert not mytree2.eq_structure(mytree)

    def test_math(self):
        mytree = nbtree.tree_simple()
        mytree.counter = 4
        # mytree.pretty_print()

        # # mytree2 = nbtree.tree_simple()
        # # mytree2.pretty_print()

        mytree + mytree
        assert mytree.counter != 8
        mytree = mytree + mytree
        assert mytree.counter == 8
        mytree = mytree - mytree
        assert mytree.counter == 0
        mytree.counter = 3
        mytree = mytree * mytree
        assert mytree.counter == 9
        with self.assertRaises(ZeroDivisionError):
            mytree / mytree
        mytree.pretty_print()

    def test_sum_inplace(self):
        mytree = nbtree.tree_simple()
        mytree.math_inplace = True
        mytree.counter = 1
        newtree = sum([mytree, mytree, mytree, mytree])
        # mytree.pretty_print()
        # Result is 8 because ever time the trees get added, their counter
        # is updated, so every new addition, the added elements are new:
        # 1+1 = 2
        # 2+2 = 4
        # 4+4 0 8
        assert mytree.counter == 8
        assert newtree.counter == 8

        mytree = nbtree.tree_simple()
        mytree.math_inplace = True
        mytree.counter = 1
        mytree = sum([mytree, mytree])
        assert mytree.counter == 2

        # test_reduce
        mytree = nbtree.tree_simple()
        mytree.math_inplace = True
        mytree.counter = 1
        mytree = mytree * 2
        mytree.pretty_print()
        assert mytree.counter == 2
        # mytree.pretty_print()

        mytree = nbtree.tree_simple()
        mytree.math_inplace = True
        mytree.counter = 1
        mytree = 2 * mytree
        # mytree.pretty_print()
        assert mytree.counter == 2

    def test_sum(self):
        mytree = nbtree.tree_simple()
        mytree.counter = 1
        newtree = sum([mytree, mytree, mytree, mytree])
        assert mytree.counter == 1
        assert newtree.counter == 4

        mytree = nbtree.tree_simple()
        mytree.counter = 1
        newtree = sum([mytree, mytree])
        assert mytree.counter == 1
        assert newtree.counter == 2

        mytree = nbtree.tree_simple()
        mytree.counter = 1
        newtree = mytree * 2
        mytree.pretty_print()
        assert mytree.counter == 1
        assert newtree.counter == 2

        mytree = nbtree.tree_simple()
        mytree.counter = 1
        newtree = 2 * mytree
        assert mytree.counter == 1
        assert newtree.counter == 2

    def test_reduce_inplace_2(self):
        import operator
        from functools import reduce

        mytree = nbtree.tree_simple()
        mytree.math_inplace = True
        mytree.counter = 2
        reduce(operator.mul, [mytree, mytree], 2)
        assert mytree.counter == 16  # 2*2, then mytree(=4)*mytree(=4) --> 16

        mytree = nbtree.tree_simple()
        mytree.math_inplace = True
        mytree.counter = 2
        reduce(operator.add, [mytree, mytree], 1)
        assert mytree.counter == 6  # 2+1, then mytree(=3) + mytree(=3) --> 6

        reduce(operator.sub, [mytree, mytree], 1)
        mytree.pretty_print()
        assert mytree.counter == 0  # 2-1, then mytree(=1) - mytree(=1) --> 0

        mytree1 = nbtree.tree_simple()
        mytree1.math_inplace = True
        mytree1.counter = 1
        mytree2 = nbtree.tree_simple()
        mytree2.math_inplace = True
        mytree2.counter = 2

        newtree = reduce(operator.mul, [mytree1, mytree2], 2)
        # mytree1.pretty_print()
        # mytree2.pretty_print()
        # newtree.pretty_print()
        assert mytree1.counter == 4
        assert mytree2.counter == 2
        assert newtree.counter == 4

    def test_reduce_inplace(self):
        import operator
        from functools import reduce

        mytree = nbtree.tree_simple()
        mytree.counter = 2
        newtree = reduce(operator.mul, [mytree, mytree], 2)
        assert mytree.counter == 2
        assert newtree.counter == 2 * 2 * 2

        mytree = nbtree.tree_simple()
        mytree.counter = 2
        newtree = reduce(operator.add, [mytree, mytree], 1)
        assert mytree.counter == 2
        assert newtree.counter == 2 + 2 + 1

        mytree = nbtree.tree_simple()
        mytree.counter = 2
        newtree = reduce(operator.sub, [mytree, mytree], 1)
        assert mytree.counter == 2
        assert newtree.counter == 1 - 2 - 2

        mytree1 = nbtree.tree_simple()
        mytree1.counter = 1
        mytree2 = nbtree.tree_simple()
        mytree2.counter = 3

        newtree = reduce(operator.mul, [mytree1, mytree2], 2)
        assert mytree1.counter == 1
        assert mytree2.counter == 3
        assert newtree.counter == 2 * 3 * 1

    def test_join_trees(self):
        import copy

        import pandas as pd

        yternary = pd.read_csv(
            os.path.join(
                TESTS_DIR, "testdata", "flowcytometry", "gated_cells", "yternary.csv"
            )
        )
        celltree = nbtree.tree_complete_cell()
        yternary_preds = celltree.predict(values=yternary)
        celltree.id_preds(yternary_preds)
        celltree.count(yternary_preds, use_ids=True)
        celltree.data = yternary

        celltree_2 = copy.deepcopy(celltree)

        newtree = celltree.join(celltree_2)

        assert celltree.data.shape[0] == 999
        assert celltree.data.shape[1] == 11
        assert celltree_2.data.shape[0] == 999
        assert celltree_2.data.shape[1] == 11
        assert newtree.data.shape[0] == 1998
        assert newtree.data.shape[1] == 11
        assert len(newtree.ids) == len(celltree.ids) * 2
        assert len(celltree_2.ids) == len(celltree.ids)

    def test_join_trees_addsource(self):
        import copy

        import pandas as pd

        yternary = pd.read_csv(
            os.path.join(
                TESTS_DIR, "testdata", "flowcytometry", "gated_cells", "yternary.csv"
            )
        )
        celltree = nbtree.tree_complete_cell()
        yternary_preds = celltree.predict(values=yternary)
        celltree.id_preds(yternary_preds)
        celltree.count(yternary_preds, use_ids=True)
        celltree.data = yternary

        celltree_2 = copy.deepcopy(celltree)

        newtree = celltree.join(celltree_2, add_source_name="source")

        assert newtree.data.columns[-1] == "source"
        print(newtree.data.columns)
        assert celltree.data.shape[0] == 999
        assert celltree.data.shape[1] == 11
        assert celltree_2.data.shape[0] == 999
        assert celltree_2.data.shape[1] == 11
        assert newtree.data.shape[0] == 1998
        assert newtree.data.shape[1] == 12
        assert len(newtree.ids) == len(celltree.ids) * 2
        assert len(celltree_2.ids) == len(celltree.ids)

        newtree = celltree.join(
            celltree_2,
            add_source_name="source",
            add_source_other="value_for_other",
            add_source_self="value_for_self",
        )
        print(newtree.data)

    def test_get_name_full(self):
        mytree = nbtree.tree_simple()
        assert mytree.get_name_full() == "/a"
        assert mytree.children[1].children[0].get_name_full() == "/a/a1/a1a"

    def test_decision_cutoff(self):
        import re

        import pandas as pd

        cellmat = pd.read_csv(
            os.path.join(
                TESTS_DIR, "testdata", "flowcytometry", "gated_cells", "cellmat.csv"
            )
        )
        cellmat.columns = [re.sub("_.*", "", x) for x in cellmat.columns]
        celltree = nbtree.tree_complete_aligned()
        a = celltree.predict(cellmat)
        print(a)

    def test_copy_structure(self):
        import pandas as pd

        yternary = pd.read_csv(
            os.path.join(
                TESTS_DIR, "testdata", "flowcytometry", "gated_cells", "yternary.csv"
            )
        )
        celltree = nbtree.tree_complete_cell()
        yternary_preds = celltree.predict(values=yternary)
        celltree.id_preds(yternary_preds)
        celltree.count(yternary_preds, use_ids=True)
        celltree.data = yternary

        new_tree = celltree.copy_structure()
        assert id(new_tree) != id(celltree)
        assert celltree.eq_structure(new_tree)
        assert new_tree.data.shape == (0, 0)
        for node in anytree.PreOrderIter(new_tree):
            assert node.counter == 0
