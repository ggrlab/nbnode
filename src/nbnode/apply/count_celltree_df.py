import anytree
import pandas as pd

from nbnode.nbnode import NBNode


def count_celltree_df(
    celltree_gated: NBNode,
) -> pd.DataFrame:
    """
        Count the number of rows in node.data (per node) grouped by
        "sample_name" and concat them into a dataframe

    Args:
        celltree_gated (NBNode):
            A gated celltree, e.g. as result from gate_csv()

    Returns:
        pd.DataFrame with
            rows:       Samples
            columns:    Nodes
            values:     Counts
            index:      Sample identifier

    Examples:
        >>> import os
        >>> from nbnode.apply.gate_csv import gate_csv
        >>> rescaled_data_dir = os.path.join(
        ...    "example_data", "asinh.align_manual.CD3_Gate"
        ...    )
        >>> if not os.path.exists(rescaled_data_dir):
        ...     raise FileNotFoundError(
        ...         rescaled_data_dir,
        ...         " has not been found, did you run tests/examples/e01_download.sh?",
        ...         )
        >>> rescaled_data_dir_files = [
        ...     os.path.join(rescaled_data_dir, file_x)
        ...     for file_x in os.listdir(rescaled_data_dir)
        ... ]
        >>> print(rescaled_data_dir_files)
        >>> a = gate_csv(csv=rescaled_data_dir_files[0])
        >>> b = gate_csv(csv=rescaled_data_dir_files[0:3])
        >>> from nbnode.apply.count_celltree import count_celltree_df
        >>> print(count_celltree_df(a))
        >>> print(count_celltree_df(b))
    """
    celltree_gated.count(use_ids=True)
    node_counts_dict = {}
    for node in anytree.PreOrderIter(celltree_gated):
        node_counts = node.data["sample_name"].value_counts()
        node_counts_dict[node.get_name_full()] = node_counts
    node_counts_df = pd.concat(node_counts_dict, axis=1)
    node_counts_df = node_counts_df.fillna(0).astype("int32")
    return node_counts_df
