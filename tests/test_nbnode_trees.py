import nbnode.nbnode_trees as nbtree


def test_trees_init():
    for x in nbtree.__dict__.keys():
        if x.startswith("tree_"):
            assert callable(nbtree.__dict__[x])
            mytree = nbtree.__dict__[x]()
            mytree.pretty_print()
