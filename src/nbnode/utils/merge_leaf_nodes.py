import pandas as pd


def merge_leaf_nodes(leaf_nodes_df: pd.DataFrame, intermediate_node: str) -> float:
    """Merge leaf node dataframe into intermediate node.

    Args:
        intermediate_node (str):
            The (full) name of the intermediate node to merge leaf nodes into.
        leaf_nodes_df (pd.DataFrame):
            The dataframe of leaf nodes to merge into the intermediate node.
            Its index are the full names of the leaf nodes. The names are organized in
            a hierarchy, separated by "/". Therefore, any intermediate node is a prefix
            of any of its leaf nodes.

            For some values it is useful to have the sum of all leaf nodes, especially
            for:

                - Dirichlet concentration parameters
                - Means of cell proportions
                - Numbers of cells

    Returns:
        float:
            Sum of all leaf nodes whose names start with the intermediate node name
    """
    return leaf_nodes_df.loc[
        [x.startswith(intermediate_node) for x in leaf_nodes_df.index]
    ].sum()
