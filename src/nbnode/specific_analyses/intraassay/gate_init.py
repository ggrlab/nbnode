import os
from typing import List, Tuple

import pandas as pd

from nbnode.apply.count_celltree_df import count_celltree_df
from nbnode.apply.gate_csv import gate_csv
from nbnode.nbnode import NBNode
from nbnode.nbnode_trees import tree_complete_aligned_v2
from nbnode.simulation.FlowSimulationTree import FlowSimulationTreeDirichlet


def gate_init(
    sample_list: List[str] = None,
) -> Tuple[NBNode, pd.DataFrame, FlowSimulationTreeDirichlet]:
    """Gate the intraassay samples and generate the Dirichlet-based simulation.

    Args:
        sample_list (List[str], optional):
            If given, only the samples with the file names indices are gated.
            Defaults to None.
    Returns:
        Tuple[NBNode, pd.DataFrame, FlowSimulationTreeDirichlet]:
            The gated celltree, the gated counts and the Dirichlet-based simulation.
    """
    # 1. Prepare the samples for gating
    rescaled_data_dir = os.path.join("example_data", "asinh.align_manual.CD3_Gate")
    if not os.path.exists(rescaled_data_dir):
        raise FileNotFoundError(
            rescaled_data_dir,
            " has not been found, did you run "
            + "tests/specific_analyses/e01_download_intraassay.sh?",
        )

    all_files = [
        os.path.join(rescaled_data_dir, file_x)
        for file_x in os.listdir(rescaled_data_dir)
    ]
    new_colnames = [
        "FS",
        "FS.0",
        "SS",
        "CD45RA",
        "CCR7",
        "CD28",
        "PD1",
        "CD27",
        "CD4",
        "CD8",
        "CD3",
        "CD57",
        "CD45",
    ]

    # 2. Mainly for testing: select only a subset of samples
    if sample_list is not None:
        all_files = [x for x in sample_list if x in all_files]

    # 3. Gate the samples
    celltree_gated = gate_csv(
        csv=all_files,
        celltree=tree_complete_aligned_v2(),
        new_colnames=new_colnames,
    )
    # celltree_gated.ids
    #   Set by celltree_gated.id_preds()
    #   Necessary to assess the population frequency distribution
    # celltree_gated.data
    #   Set by celltree_gated.data = data_all
    #   Necessary to estimate the distribution of cells in marker values PER POPULATION
    #

    # 4. Generate the Dirichlet-based simulation
    flowsim_tree = FlowSimulationTreeDirichlet(
        rootnode=celltree_gated,
        include_features=new_colnames,
        node_percentages=None,
        data_cellgroup_col="sample_name",  # is the default
    )

    # 5. Extract the gated counts
    node_counts_df = count_celltree_df(celltree_gated)

    return celltree_gated, node_counts_df, flowsim_tree


if __name__ == "__main__":
    from nbnode.io.pickle_open_dump import pickle_open_dump

    celltree, node_counts_df, flow_dist = gate_init()

    pickle_open_dump(
        flow_dist, "FlowSimulationTreeDirichlet.DuraCloneTcell.CD3.rescaled.pickle"
    )
