def test_flatten():
    from nbnode_pyscaffold.plot.utils import flatten

    assert [] == flatten([[[[]]], [], [[]], [[], []]])  # empty multidimensional list
    assert [1, 2, 3, 4, 5, 6, 7, 8] == flatten(
        [[1], [2, 3], [4, [5, [6, [7, [8]]]]]]
    )  # multiple nested list
    assert [1, 2, 3, "4", {5: 5}] == flatten([[1, [[2]], [[[3]]]], [["4"], {5: 5}]])

def test_plot_save_unified():
    # Is used all around in test_NBNode.py
    pass