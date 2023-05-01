import anytree

from nbnode.nbnode import NBNode


def tree_simple() -> NBNode:
    """Simple tree for testing.

    Returns:
        NBNode::

            a (counter:0, decision_name:None, decision_value:None)
            ├── a0 (counter:0, decision_name:m1, decision_value:-1)
            ├── a1 (counter:0, decision_name:m1, decision_value:1)
            │   └── a1a (counter:0, decision_name:m2, decision_value:test)
            └── a2 (counter:0, decision_name:m3, decision_value:another)

    """
    mytree = NBNode("a")
    # a0 =
    NBNode("a0", parent=mytree, decision_value=-1, decision_name="m1")
    a1 = NBNode("a1", parent=mytree, decision_value=1, decision_name="m1")
    # a2 =
    NBNode("a2", parent=mytree, decision_value="another", decision_name="m3")
    # a1a =
    NBNode("a1a", parent=a1, decision_value="test", decision_name="m2")
    # a0, a1, a2, a1a = a0, a1, a2, a1a  # make flake happy
    return mytree


def tree_simpleB() -> NBNode:
    """Another simple tree for testing.

    Returns:
        NBNode::

            a (counter:0, decision_name:None, decision_value:None)
            ├── a0 (counter:0, decision_name:m1, decision_value:-1)
            ├── a1 (counter:0, decision_name:m1, decision_value:1)
            │   ├── a1a (counter:0, decision_name:m2, decision_value:test)
            │   └── a1b (counter:0, decision_name:m2, decision_value:tmp)
            └── a2 (counter:0, decision_name:m3, decision_value:another)
    """
    mytree = NBNode("a")
    # a0 =
    NBNode("a0", parent=mytree, decision_value=-1, decision_name="m1")
    a1 = NBNode("a1", parent=mytree, decision_value=1, decision_name="m1")
    # a2 =
    NBNode("a2", parent=mytree, decision_value="another", decision_name="m3")
    # a1a =
    NBNode("a1a", parent=a1, decision_value="test", decision_name="m2")
    NBNode("a1b", parent=mytree["/a/a1"], decision_value="tmp", decision_name="m2")
    return mytree


def tree_complex() -> NBNode:
    """Complex tree to use with yternary.

    Returns:
        NBNode::

            AllCells (counter:0, decision_name:None, decision_value:None)
            ├── not CD45 (counter:0, decision_name:CD45, decision_value:-1)
            └── CD45+ (counter:0, decision_name:CD45, decision_value:1)
                ├── not CD3 (counter:0, decision_name:CD3, decision_value:-1)
                │   ├── not MNC (counter:0, decision_name:MNC, decision_value:-1)
                │   └── MNCs (counter:0, decision_name:MNC, decision_value:1)
                │       ├── other (counter:0, decision_name:CD4, decision_value:-1)
                │       └── Monocytes (counter:0, decision_name:CD4, decision_value:1)
                └── CD3+ (counter:0, decision_name:CD3, decision_value:1)
                    ├── DN (counter:0, decision_name:['CD4', 'CD8'],
                            decision_value:[-1, -1])
                    ├── DP (counter:0, decision_name:['CD4', 'CD8'],
                            decision_value:[1, 1])
                    ├── CD4-/CD8+ (counter:0, decision_name:['CD4', 'CD8'],
                                    decision_value:[-1, 1])
                    └── CD4+/CD8- (counter:0, decision_name:['CD4', 'CD8'],
                                    decision_value:[1, -1])
    """
    celltree = NBNode(
        "AllCells",
        children=[
            NBNode("not CD45", decision_name="CD45", decision_value=-1, children=None),
            NBNode(
                "CD45+",
                decision_name="CD45",
                decision_value=1,
                children=[
                    NBNode(
                        "not CD3",
                        decision_name="CD3",
                        decision_value=-1,
                        children=[
                            NBNode(
                                "not MNC",
                                decision_name="MNC",
                                decision_value=-1,
                                children=None,
                            ),
                            NBNode(
                                "MNCs",
                                decision_name="MNC",
                                decision_value=1,
                                children=[
                                    NBNode(
                                        "other", decision_name="CD4", decision_value=-1
                                    ),
                                    NBNode(
                                        "Monocytes",
                                        decision_name="CD4",
                                        decision_value=1,
                                    ),
                                ],
                            ),
                        ],
                    ),
                    NBNode(
                        "CD3+",
                        decision_name="CD3",
                        decision_value=1,
                        children=[
                            NBNode(
                                "DN",
                                decision_name=["CD4", "CD8"],
                                decision_value=[-1, -1],
                            ),
                            NBNode(
                                "DP",
                                decision_name=["CD4", "CD8"],
                                decision_value=[1, 1],
                            ),
                            NBNode(
                                "CD4-/CD8+",
                                decision_name=["CD4", "CD8"],
                                decision_value=[-1, 1],
                            ),
                            NBNode(
                                "CD4+/CD8-",
                                decision_name=["CD4", "CD8"],
                                decision_value=[1, -1],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )
    return celltree


def tree_complete_cell() -> NBNode:
    """Complete tree for T-cell panel of Beckman Coulter.

    Returns:
        NBNode::

            AllCells ()
            ├── not CD45 ()
            └── CD45+ ()
                ├── not CD3 ()
                │   ├── not MNC ()
                │   └── MNCs ()
                │       ├── other ()
                │       └── Monocytes ()
                └── CD3+ ()
                    ├── DN ()
                    ├── DP ()
                    ├── CD4-/CD8+ ()
                    │   ├── naive ()
                    │   │   ├── CD27+/CD28+ ()
                    │   │   │   ├── CD57+/PD1+ ()
                    │   │   │   ├── CD57+/PD1- ()
                    │   │   │   ├── CD57-/PD1+ ()
                    │   │   │   └── CD57-/PD1- ()
                    │   │   ├── CD27+/CD28- ()
                    │   │   │   ├── CD57+/PD1+ ()
                    │   │   │   ├── CD57+/PD1- ()
                    │   │   │   ├── CD57-/PD1+ ()
                    │   │   │   └── CD57-/PD1- ()
                    │   │   ├── CD27-/CD28+ ()
                    │   │   │   ├── CD57+/PD1+ ()
                    │   │   │   ├── CD57+/PD1- ()
                    │   │   │   ├── CD57-/PD1+ ()
                    │   │   │   └── CD57-/PD1- ()
                    │   │   └── CD27-/CD28- ()
                    │   │       ├── CD57+/PD1+ ()
                    │   │       ├── CD57+/PD1- ()
                    │   │       ├── CD57-/PD1+ ()
                    │   │       └── CD57-/PD1- ()
                    │   ├── Tcm ()
                    │   │   ├── CD27+/CD28+ ()
                    │   │   │   ├── CD57+/PD1+ ()
                    │   │   │   ├── CD57+/PD1- ()
                    │   │   │   ├── CD57-/PD1+ ()
                    │   │   │   └── CD57-/PD1- ()
                    │   │   ├── CD27+/CD28- ()
                    │   │   │   ├── CD57+/PD1+ ()
                    │   │   │   ├── CD57+/PD1- ()
                    │   │   │   ├── CD57-/PD1+ ()
                    │   │   │   └── CD57-/PD1- ()
                    │   │   ├── CD27-/CD28+ ()
                    │   │   │   ├── CD57+/PD1+ ()
                    │   │   │   ├── CD57+/PD1- ()
                    │   │   │   ├── CD57-/PD1+ ()
                    │   │   │   └── CD57-/PD1- ()
                    │   │   └── CD27-/CD28- ()
                    │   │       ├── CD57+/PD1+ ()
                    │   │       ├── CD57+/PD1- ()
                    │   │       ├── CD57-/PD1+ ()
                    │   │       └── CD57-/PD1- ()
                    │   ├── Temra ()
                    │   │   ├── CD27+/CD28+ ()
                    │   │   │   ├── CD57+/PD1+ ()
                    │   │   │   ├── CD57+/PD1- ()
                    │   │   │   ├── CD57-/PD1+ ()
                    │   │   │   └── CD57-/PD1- ()
                    │   │   ├── CD27+/CD28- ()
                    │   │   │   ├── CD57+/PD1+ ()
                    │   │   │   ├── CD57+/PD1- ()
                    │   │   │   ├── CD57-/PD1+ ()
                    │   │   │   └── CD57-/PD1- ()
                    │   │   ├── CD27-/CD28+ ()
                    │   │   │   ├── CD57+/PD1+ ()
                    │   │   │   ├── CD57+/PD1- ()
                    │   │   │   ├── CD57-/PD1+ ()
                    │   │   │   └── CD57-/PD1- ()
                    │   │   └── CD27-/CD28- ()
                    │   │       ├── CD57+/PD1+ ()
                    │   │       ├── CD57+/PD1- ()
                    │   │       ├── CD57-/PD1+ ()
                    │   │       └── CD57-/PD1- ()
                    │   └── Tem ()
                    │       ├── CD27+/CD28+ ()
                    │       │   ├── CD57+/PD1+ ()
                    │       │   ├── CD57+/PD1- ()
                    │       │   ├── CD57-/PD1+ ()
                    │       │   └── CD57-/PD1- ()
                    │       ├── CD27+/CD28- ()
                    │       │   ├── CD57+/PD1+ ()
                    │       │   ├── CD57+/PD1- ()
                    │       │   ├── CD57-/PD1+ ()
                    │       │   └── CD57-/PD1- ()
                    │       ├── CD27-/CD28+ ()
                    │       │   ├── CD57+/PD1+ ()
                    │       │   ├── CD57+/PD1- ()
                    │       │   ├── CD57-/PD1+ ()
                    │       │   └── CD57-/PD1- ()
                    │       └── CD27-/CD28- ()
                    │           ├── CD57+/PD1+ ()
                    │           ├── CD57+/PD1- ()
                    │           ├── CD57-/PD1+ ()
                    │           └── CD57-/PD1- ()
                    └── CD4+/CD8- ()
                        ├── naive ()
                        │   ├── CD27+/CD28+ ()
                        │   │   ├── CD57+/PD1+ ()
                        │   │   ├── CD57+/PD1- ()
                        │   │   ├── CD57-/PD1+ ()
                        │   │   └── CD57-/PD1- ()
                        │   ├── CD27+/CD28- ()
                        │   │   ├── CD57+/PD1+ ()
                        │   │   ├── CD57+/PD1- ()
                        │   │   ├── CD57-/PD1+ ()
                        │   │   └── CD57-/PD1- ()
                        │   ├── CD27-/CD28+ ()
                        │   │   ├── CD57+/PD1+ ()
                        │   │   ├── CD57+/PD1- ()
                        │   │   ├── CD57-/PD1+ ()
                        │   │   └── CD57-/PD1- ()
                        │   └── CD27-/CD28- ()
                        │       ├── CD57+/PD1+ ()
                        │       ├── CD57+/PD1- ()
                        │       ├── CD57-/PD1+ ()
                        │       └── CD57-/PD1- ()
                        ├── Tcm ()
                        │   ├── CD27+/CD28+ ()
                        │   │   ├── CD57+/PD1+ ()
                        │   │   ├── CD57+/PD1- ()
                        │   │   ├── CD57-/PD1+ ()
                        │   │   └── CD57-/PD1- ()
                        │   ├── CD27+/CD28- ()
                        │   │   ├── CD57+/PD1+ ()
                        │   │   ├── CD57+/PD1- ()
                        │   │   ├── CD57-/PD1+ ()
                        │   │   └── CD57-/PD1- ()
                        │   ├── CD27-/CD28+ ()
                        │   │   ├── CD57+/PD1+ ()
                        │   │   ├── CD57+/PD1- ()
                        │   │   ├── CD57-/PD1+ ()
                        │   │   └── CD57-/PD1- ()
                        │   └── CD27-/CD28- ()
                        │       ├── CD57+/PD1+ ()
                        │       ├── CD57+/PD1- ()
                        │       ├── CD57-/PD1+ ()
                        │       └── CD57-/PD1- ()
                        ├── Temra ()
                        │   ├── CD27+/CD28+ ()
                        │   │   ├── CD57+/PD1+ ()
                        │   │   ├── CD57+/PD1- ()
                        │   │   ├── CD57-/PD1+ ()
                        │   │   └── CD57-/PD1- ()
                        │   ├── CD27+/CD28- ()
                        │   │   ├── CD57+/PD1+ ()
                        │   │   ├── CD57+/PD1- ()
                        │   │   ├── CD57-/PD1+ ()
                        │   │   └── CD57-/PD1- ()
                        │   ├── CD27-/CD28+ ()
                        │   │   ├── CD57+/PD1+ ()
                        │   │   ├── CD57+/PD1- ()
                        │   │   ├── CD57-/PD1+ ()
                        │   │   └── CD57-/PD1- ()
                        │   └── CD27-/CD28- ()
                        │       ├── CD57+/PD1+ ()
                        │       ├── CD57+/PD1- ()
                        │       ├── CD57-/PD1+ ()
                        │       └── CD57-/PD1- ()
                        └── Tem ()
                            ├── CD27+/CD28+ ()
                            │   ├── CD57+/PD1+ ()
                            │   ├── CD57+/PD1- ()
                            │   ├── CD57-/PD1+ ()
                            │   └── CD57-/PD1- ()
                            ├── CD27+/CD28- ()
                            │   ├── CD57+/PD1+ ()
                            │   ├── CD57+/PD1- ()
                            │   ├── CD57-/PD1+ ()
                            │   └── CD57-/PD1- ()
                            ├── CD27-/CD28+ ()
                            │   ├── CD57+/PD1+ ()
                            │   ├── CD57+/PD1- ()
                            │   ├── CD57-/PD1+ ()
                            │   └── CD57-/PD1- ()
                            └── CD27-/CD28- ()
                                ├── CD57+/PD1+ ()
                                ├── CD57+/PD1- ()
                                ├── CD57-/PD1+ ()
                                └── CD57-/PD1- ()
    """
    celltree = tree_complex()

    temra_part = [
        NBNode("naive", decision_name=["CCR7", "CD45RA"], decision_value=[1, 1]),
        NBNode("Tcm", decision_name=["CCR7", "CD45RA"], decision_value=[1, -1]),
        NBNode("Temra", decision_name=["CCR7", "CD45RA"], decision_value=[-1, 1]),
        NBNode("Tem", decision_name=["CCR7", "CD45RA"], decision_value=[-1, -1]),
    ]
    post_temra_1_part = [
        NBNode("CD27+/CD28+", decision_name=["CD27", "CD28"], decision_value=[1, 1]),
        NBNode("CD27+/CD28-", decision_name=["CD27", "CD28"], decision_value=[1, -1]),
        NBNode("CD27-/CD28+", decision_name=["CD27", "CD28"], decision_value=[-1, 1]),
        NBNode("CD27-/CD28-", decision_name=["CD27", "CD28"], decision_value=[-1, -1]),
    ]
    post_temra_2_part = [
        NBNode("CD57+/PD1+", decision_name=["CD57", "PD1"], decision_value=[1, 1]),
        NBNode("CD57+/PD1-", decision_name=["CD57", "PD1"], decision_value=[1, -1]),
        NBNode("CD57-/PD1+", decision_name=["CD57", "PD1"], decision_value=[-1, 1]),
        NBNode("CD57-/PD1-", decision_name=["CD57", "PD1"], decision_value=[-1, -1]),
    ]

    for post1_node in post_temra_1_part:
        post1_node.insert_nodes(post_temra_2_part, copy_list=True)
    for temra_node in temra_part:
        temra_node.insert_nodes(post_temra_1_part, copy_list=True)
    for hutch_node in anytree.findall(
        celltree, filter_=lambda node: node.name in ["CD4-/CD8+", "CD4+/CD8-"]
    ):
        hutch_node.insert_nodes(temra_part, copy_list=True)
    return celltree


def tree_simple_cutoff_NOTWORKING() -> NBNode:
    """Not working simple tree with decision cutoffs, only for testing.

    Blank

    Returns:
        NBNode::

            a (counter:0, decision_name:None, decision_value:None)
            ├── a0 (counter:0, decision_name:m1, decision_value:-1)
            ├── a1 (counter:0, decision_name:m1, decision_value:1)
            │   └── a1a (counter:0, decision_name:m2, decision_value:test)
            ├── a2 (counter:0, decision_name:m3, decision_value:another)
            └── a3 (counter:0, decision_name:['m1', 'm4'], decision_value:[0, 1])
    """
    mytree = NBNode("a")
    # a0 =
    NBNode(
        "a0", parent=mytree, decision_value=-1, decision_cutoff=0, decision_name="m1"
    )
    a1 = NBNode(
        "a1", parent=mytree, decision_value=1, decision_cutoff=0, decision_name="m1"
    )
    # a2 =
    NBNode("a2", parent=mytree, decision_value="another", decision_name="m3")
    # a1a =
    NBNode("a1a", parent=a1, decision_value="test", decision_name="m2")
    NBNode(
        "a3",
        parent=mytree,
        decision_value=[0, 1],
        decision_name=["m1", "m4"],
        decision_cutoff=[0, 0],
    )
    return mytree


def tree_simple_cutoff() -> NBNode:
    """_summary_.

    Returns:
        NBNode::

            a (counter:0, decision_name:None, decision_value:None)
            ├── a0 (counter:0, decision_name:m1, decision_value:-1)
            ├── a1 (counter:0, decision_name:m1, decision_value:1)
            │   └── a1a (counter:0, decision_name:m2, decision_value:test)
            └── a2 (counter:0, decision_name:m3, decision_value:another)
    """
    mytree = NBNode("a")
    # a0 =
    NBNode(
        "a0", parent=mytree, decision_value=-1, decision_cutoff=0, decision_name="m1"
    )
    a1 = NBNode(
        "a1", parent=mytree, decision_value=1, decision_cutoff=0, decision_name="m1"
    )
    # a2 =
    NBNode("a2", parent=mytree, decision_value="another", decision_name="m3")
    # a1a =
    NBNode("a1a", parent=a1, decision_value="test", decision_name="m2")
    return mytree


def tree_simple_cutoff_mixed() -> NBNode:
    """Functioning tree with decision cutoffs, testing.

    Returns:
        NBNode::

            a (counter:0, decision_name:None, decision_value:None)
            ├── a0 (counter:0, decision_name:m1, decision_value:-1)
            ├── a1 (counter:0, decision_name:m1, decision_value:1)
            │   └── a1a (counter:0, decision_name:m2, decision_value:test)
            ├── a2 (counter:0, decision_name:m3, decision_value:another)
            └── a3 (counter:0, decision_name:['m2', 'm4'], decision_value:['test', 1])
    """
    mytree = NBNode("a")
    # a0 =
    NBNode("a0", parent=mytree, decision_value=-1, decision_name="m1")
    a1 = NBNode("a1", parent=mytree, decision_value=1, decision_name="m1")
    # a2 =
    NBNode("a2", parent=mytree, decision_value="another", decision_name="m3")
    # a1a =
    NBNode("a1a", parent=a1, decision_value="test", decision_name="m2")
    NBNode(
        "a3",
        parent=mytree,
        decision_value=["test", 1],
        decision_name=["m2", "m4"],
        decision_cutoff=[None, 0],
    )
    return mytree


def tree_complete_aligned() -> NBNode:
    """Tree for the aligned (now "rescaled") T-cell panel data.

    Uses decision cutoffs.

    Returns:
        NBNode: See ``tree_complete_cell()``, only the decision cutoffs are different
    """
    # Betreff:	gates
    # Erstellt von:	Jxxx.Hxxx@ukr.de
    # Geplantes Datum:
    # Erstellungsdatum:	22.09.2022, 16:23
    # Von:	JH
    # An: G G (Gxx.Gxx@ukr.de)
    celltree = tree_complex()

    celltree = NBNode(
        "AllCells",
        children=[
            NBNode(
                "DN",
                decision_name=["CD4", "CD8"],
                decision_value=[-1, -1],
                decision_cutoff=[0.19, 0.2],
            ),
            NBNode(
                "DP",
                decision_name=["CD4", "CD8"],
                decision_value=[1, 1],
                decision_cutoff=[0.19, 0.2],
            ),
            NBNode(
                "CD4-/CD8+",
                decision_name=["CD4", "CD8"],
                decision_value=[-1, 1],
                decision_cutoff=[0.19, 0.2],
            ),
            NBNode(
                "CD4+/CD8-",
                decision_name=["CD4", "CD8"],
                decision_value=[1, -1],
                decision_cutoff=[0.19, 0.2],
            ),
        ],
    )

    temra_part = [
        NBNode(
            "naive",
            decision_name=["CCR7", "CD45RA"],
            decision_value=[1, 1],
            decision_cutoff=[0.24, 0.12],
        ),
        NBNode(
            "Tcm",
            decision_name=["CCR7", "CD45RA"],
            decision_value=[1, -1],
            decision_cutoff=[0.24, 0.12],
        ),
        NBNode(
            "Temra",
            decision_name=["CCR7", "CD45RA"],
            decision_value=[-1, 1],
            decision_cutoff=[0.24, 0.12],
        ),
        NBNode(
            "Tem",
            decision_name=["CCR7", "CD45RA"],
            decision_value=[-1, -1],
            decision_cutoff=[0.24, 0.12],
        ),
    ]
    post_temra_1_part = [
        NBNode(
            "CD27+/CD28+",
            decision_name=["CD27", "CD28"],
            decision_value=[1, 1],
            decision_cutoff=[0.36, 0.30],
        ),
        NBNode(
            "CD27+/CD28-",
            decision_name=["CD27", "CD28"],
            decision_value=[1, -1],
            decision_cutoff=[0.36, 0.30],
        ),
        NBNode(
            "CD27-/CD28+",
            decision_name=["CD27", "CD28"],
            decision_value=[-1, 1],
            decision_cutoff=[0.36, 0.30],
        ),
        NBNode(
            "CD27-/CD28-",
            decision_name=["CD27", "CD28"],
            decision_value=[-1, -1],
            decision_cutoff=[0.36, 0.30],
        ),
    ]
    post_temra_2_part = [
        NBNode(
            "CD57+/PD1+",
            decision_name=["CD57", "PD1"],
            decision_value=[1, 1],
            decision_cutoff=[0.23, 0.52],
        ),
        NBNode(
            "CD57+/PD1-",
            decision_name=["CD57", "PD1"],
            decision_value=[1, -1],
            decision_cutoff=[0.23, 0.52],
        ),
        NBNode(
            "CD57-", decision_name=["CD57"], decision_value=[-1], decision_cutoff=[0.23]
        ),
    ]

    for post1_node in post_temra_1_part:
        post1_node.insert_nodes(post_temra_2_part, copy_list=True)
    for temra_node in temra_part:
        temra_node.insert_nodes(post_temra_1_part, copy_list=True)
    for hutch_node in anytree.findall(
        celltree, filter_=lambda node: node.name in ["CD4-/CD8+", "CD4+/CD8-"]
    ):
        hutch_node.insert_nodes(temra_part, copy_list=True)
    return celltree


def tree_complete_aligned_v2():
    """
    Tree for the aligned (now "rescaled") T-cell panel data.

    Uses decision cutoffs.

    Returns:
        NBNode: See ``tree_complete_aligned()``, only the decision cutoffs are different
    """
    # Given to GG by JH per paper.
    # Essentially the exact same gating hierarchy as tree_complete_aligned_v2 with
    # really minor changes
    # CD4, CD8 cutoffs remain
    # CD4:      0.20 --> 0.20   UNCHANGED
    # CD8:      0.19 --> 0.19   UNCHANGED
    # CD45RA:   0.12 --> 0.20
    # CCR7:     0.24 --> 0.15
    # CD27:     0.36 --> 0.30
    # CD28:     0.30 --> 0.30   UNCHANGED
    # CD57:     0.23 --> 0.23   UNCHANGED
    # PD1:      0.52 --> 0.52   UNCHANGED
    celltree = tree_complex()

    celltree = NBNode(
        "AllCells",
        children=[
            NBNode(
                "DN",
                decision_name=["CD4", "CD8"],
                decision_value=[-1, -1],
                decision_cutoff=[0.19, 0.2],
            ),
            NBNode(
                "DP",
                decision_name=["CD4", "CD8"],
                decision_value=[1, 1],
                decision_cutoff=[0.19, 0.2],
            ),
            NBNode(
                "CD4-/CD8+",
                decision_name=["CD4", "CD8"],
                decision_value=[-1, 1],
                decision_cutoff=[0.19, 0.2],
            ),
            NBNode(
                "CD4+/CD8-",
                decision_name=["CD4", "CD8"],
                decision_value=[1, -1],
                decision_cutoff=[0.19, 0.2],
            ),
        ],
    )

    temra_part = [
        NBNode(
            "naive",
            decision_name=["CCR7", "CD45RA"],
            decision_value=[1, 1],
            decision_cutoff=[0.15, 0.20],
        ),
        NBNode(
            "Tcm",
            decision_name=["CCR7", "CD45RA"],
            decision_value=[1, -1],
            decision_cutoff=[0.15, 0.20],
        ),
        NBNode(
            "Temra",
            decision_name=["CCR7", "CD45RA"],
            decision_value=[-1, 1],
            decision_cutoff=[0.15, 0.20],
        ),
        NBNode(
            "Tem",
            decision_name=["CCR7", "CD45RA"],
            decision_value=[-1, -1],
            decision_cutoff=[0.15, 0.20],
        ),
    ]
    post_temra_1_part = [
        NBNode(
            "CD27+/CD28+",
            decision_name=["CD27", "CD28"],
            decision_value=[1, 1],
            decision_cutoff=[0.30, 0.30],
        ),
        NBNode(
            "CD27+/CD28-",
            decision_name=["CD27", "CD28"],
            decision_value=[1, -1],
            decision_cutoff=[0.30, 0.30],
        ),
        NBNode(
            "CD27-/CD28+",
            decision_name=["CD27", "CD28"],
            decision_value=[-1, 1],
            decision_cutoff=[0.30, 0.30],
        ),
        NBNode(
            "CD27-/CD28-",
            decision_name=["CD27", "CD28"],
            decision_value=[-1, -1],
            decision_cutoff=[0.30, 0.30],
        ),
    ]
    post_temra_2_part = [
        NBNode(
            "CD57+/PD1+",
            decision_name=["CD57", "PD1"],
            decision_value=[1, 1],
            decision_cutoff=[0.23, 0.52],
        ),
        NBNode(
            "CD57+/PD1-",
            decision_name=["CD57", "PD1"],
            decision_value=[1, -1],
            decision_cutoff=[0.23, 0.52],
        ),
        NBNode(
            "CD57-", decision_name=["CD57"], decision_value=[-1], decision_cutoff=[0.23]
        ),
    ]

    for post1_node in post_temra_1_part:
        post1_node.insert_nodes(post_temra_2_part, copy_list=True)
    for temra_node in temra_part:
        temra_node.insert_nodes(post_temra_1_part, copy_list=True)
    for hutch_node in anytree.findall(
        celltree, filter_=lambda node: node.name in ["CD4-/CD8+", "CD4+/CD8-"]
    ):
        hutch_node.insert_nodes(temra_part, copy_list=True)
    return celltree


def tree_complete_aligned_trunk() -> NBNode:
    """Trunk of the tree_complete_aligned_v2 tree.

    Returns:
        NBNode::

            AllCells (counter:0, decision_name:None, decision_value:None)
            ├── DN (counter:0, decision_name:['CD4', 'CD8'], decision_value:[-1, -1])
            ├── DP (counter:0, decision_name:['CD4', 'CD8'], decision_value:[1, 1])
            ├── CD4-/CD8+ (counter:0, decision_name:['CD4', 'CD8'],
                            decision_value:[-1, 1])
            │   ├── naive (counter:0, decision_name:['CCR7', 'CD45RA'],
                            decision_value:[1, 1])
            │   ├── Tcm (counter:0, decision_name:['CCR7', 'CD45RA'],
                            decision_value:[1, -1])
            │   ├── Temra (counter:0, decision_name:['CCR7', 'CD45RA'],
                            decision_value:[-1, 1])
            │   └── Tem (counter:0, decision_name:['CCR7', 'CD45RA'],
                            decision_value:[-1, -1])
            └── CD4+/CD8- (counter:0, decision_name:['CD4', 'CD8'],
                            decision_value:[1, -1])
                ├── naive (counter:0, decision_name:['CCR7', 'CD45RA'],
                        decision_value:[1, 1])
                ├── Tcm (counter:0, decision_name:['CCR7', 'CD45RA'],
                        decision_value:[1, -1])
                ├── Temra (counter:0, decision_name:['CCR7', 'CD45RA'],
                        decision_value:[-1, 1])
                └── Tem (counter:0, decision_name:['CCR7', 'CD45RA'],
                        decision_value:[-1, -1])

    """
    # Betreff:	gates
    # Erstellt von:	Jxxx.Hxxx@ukr.de
    # Geplantes Datum:
    # Erstellungsdatum:	22.09.2022, 16:23
    # Von:	JH
    # An: G G (Gxx.Gxx@ukr.de)

    celltree = NBNode(
        "AllCells",
        children=[
            NBNode(
                "DN",
                decision_name=["CD4", "CD8"],
                decision_value=[-1, -1],
                decision_cutoff=[0.19, 0.2],
            ),
            NBNode(
                "DP",
                decision_name=["CD4", "CD8"],
                decision_value=[1, 1],
                decision_cutoff=[0.19, 0.2],
            ),
            NBNode(
                "CD4-/CD8+",
                decision_name=["CD4", "CD8"],
                decision_value=[-1, 1],
                decision_cutoff=[0.19, 0.2],
            ),
            NBNode(
                "CD4+/CD8-",
                decision_name=["CD4", "CD8"],
                decision_value=[1, -1],
                decision_cutoff=[0.19, 0.2],
            ),
        ],
    )

    temra_part = [
        NBNode(
            "naive",
            decision_name=["CCR7", "CD45RA"],
            decision_value=[1, 1],
            decision_cutoff=[0.24, 0.12],
        ),
        NBNode(
            "Tcm",
            decision_name=["CCR7", "CD45RA"],
            decision_value=[1, -1],
            decision_cutoff=[0.24, 0.12],
        ),
        NBNode(
            "Temra",
            decision_name=["CCR7", "CD45RA"],
            decision_value=[-1, 1],
            decision_cutoff=[0.24, 0.12],
        ),
        NBNode(
            "Tem",
            decision_name=["CCR7", "CD45RA"],
            decision_value=[-1, -1],
            decision_cutoff=[0.24, 0.12],
        ),
    ]
    for hutch_node in anytree.findall(
        celltree, filter_=lambda node: node.name in ["CD4-/CD8+", "CD4+/CD8-"]
    ):
        hutch_node.insert_nodes(temra_part, copy_list=True)
    return celltree
