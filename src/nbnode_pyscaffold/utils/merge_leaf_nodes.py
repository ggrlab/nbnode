import pandas as pd

from nbnode_pyscaffold.nbnode import NBNode


def merge_leaf_nodes(intermediate_node: NBNode, leaf_nodes_df: pd.DataFrame) -> float:
    return leaf_nodes_df.loc[
        [x.startswith(intermediate_node) for x in leaf_nodes_df.index]
    ].sum()
