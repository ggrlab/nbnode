def test_flatten():
    from nbnode.plot.utils import flatten

    assert [] == flatten([[[[]]], [], [[]], [[], []]])  # empty multidimensional list
    assert [1, 2, 3, 4, 5, 6, 7, 8] == flatten(
        [[1], [2, 3], [4, [5, [6, [7, [8]]]]]]
    )  # multiple nested list
    assert [1, 2, 3, "4", {5: 5}] == flatten([[1, [[2]], [[[3]]]], [["4"], {5: 5}]])
    assert flatten([[1, 2, 3, 4, 5, 6, 7, 8]]) == [1, 2, 3, 4, 5, 6, 7, 8]


def test_plot_save_unified_dot():
    import os

    import anytree.exporter as a_exp
    import pydotplus

    import nbnode.nbnode_trees as nbtree
    from nbnode.plot.utils import plot_save_unified

    simpletree = nbtree.tree_simple()
    dot_data = a_exp.UniqueDotExporter(
        simpletree,
        options=['node [shape=box, style="filled", color="black"];'],
        nodeattrfunc=lambda node: 'label="{}", fillcolor="white"'.format(node.name),
    )
    dotdata_str = "\n".join([x for x in dot_data])
    # print(dotdata_str)

    os.makedirs("tests_output", exist_ok=True)
    graph: pydotplus.Dot = pydotplus.graph_from_dot_data(dotdata_str)
    plot_save_unified(
        any_plot=graph, file="tests_output/pydotplus_graph_nocolor_NOTpdf.NOTpdf"
    )
    assert os.path.exists("tests_output/pydotplus_graph_nocolor_NOTpdf.pdf")

    plot_save_unified(any_plot=graph, file="tests_output/pydotplus_graph_nocolor.pdf")


def test_plot_save_unified_matplotlib():
    import os

    import matplotlib.pyplot as plt

    from nbnode.plot.utils import plot_save_unified

    os.makedirs("tests_output", exist_ok=True)
    plt.plot([1, 2, 3, 4])
    plot_save_unified(any_plot=plt, file="tests_output/matplotlib_graph.png")
    assert os.path.exists("tests_output/matplotlib_graph.png")
    plt.close("all")
