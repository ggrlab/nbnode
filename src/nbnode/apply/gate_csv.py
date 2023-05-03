from typing import List, Union

import pandas as pd

import nbnode.nbnode_trees as nbtrees


def gate_csv(
    csv: Union[str, List[str]],
    celltree="default",
    new_colnames=[
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
    ],
    verbose=True,
):
    """_summary_

    Args:
        csv (Union[str, List[str]]):
            Path to a single .csv file or a list of .csv files
        celltree (str, optional): _description_. Defaults to "default".
        new_colnames (List[str]):
            vector of new colnames for the csvs. Usually used to make the colnames of
            the csvs matching to the celltree.
        verbose (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_

    Examples:
        >>> rescaled_data_dir = os.path.join(
        ...     "example_data", "asinh.align_manual.CD3_Gate")
        >>> if not os.path.exists(rescaled_data_dir):
        ...     raise FileNotFoundError(
        ...         rescaled_data_dir,
        ...         " has not been found, did you run tests/examples/e01_download.sh?",
        ...         )
        >>> rescaled_data_dir_files = [os.path.join(rescaled_data_dir, file_x)
        ... for file_x in os.listdir(rescaled_data_dir)]
        >>> print(rescaled_data_dir_files)
        >>> gate_csv(csv=rescaled_data_dir_files[0])
        >>> gate_csv(csv=rescaled_data_dir_files[0:3])
    """
    if celltree == "default":
        celltree = nbtrees.tree_complete_aligned_v2()

    # If the csv is a single file, it is of instance string,
    # therefore encase it inside a list to iterate over
    if isinstance(csv, str):
        csv = [csv]

    data_all = None
    predicted_nodes_all = []
    for path_x in csv:
        # path_x = os.path.join(rescaled_data_dir, file_x)
        print(path_x)
        # Read the csv
        loaded_csv = pd.read_csv(path_x)
        # Rename the columns such that they match the names of celltree
        if new_colnames is not None:
            loaded_csv.columns = new_colnames
        # The name of the sample is necessary later in FlowSimulationTreeDirichlet
        loaded_csv["sample_name"] = path_x

        # Per row of the loaded csv, get the end node of the celltree
        predicted_nodes = celltree.predict(values=loaded_csv)
        predicted_nodes_all += list(predicted_nodes)
        if data_all is None:
            data_all = loaded_csv
        else:
            data_all = pd.concat([data_all, loaded_csv])

    # data_all
    #   contains all loaded csvs rbound
    # predicted_nodes_all
    #   contains all predicted end nodes in the same order as data_all
    #  (first row of data_all is the first element of predicted_nodes_all)
    if verbose:
        print("    Set celltree.id_preds() based on predicted_nodes_all")

    # Given a list of nodes, enumerate through them and assign this
    # (`enumerate(node_list)`)
    # number to self.ids.
    celltree.id_preds(predicted_nodes_all, reset_ids=True)

    if verbose:
        print("    Count cells based on ids. celltree.count(use_ids=True)")
    celltree.count(use_ids=True)

    if verbose:
        print("    Set celltree.data")
    celltree.data = data_all
    return celltree
