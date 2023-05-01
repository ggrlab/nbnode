# import datatable
# import re

# import nbnode.nbnode_trees as nbtrees
# import torch
# import numpy as np

# from ccc.datasets.InMemoryPointCloud import InMemoryPointCloud


# def gate_pointcloud(
#     impc, celltree="default", manual_seed_impc_transform=39486, verbose=True,
# ):
#     """_summary_

#     Args:
#         impc (InMemoryPointCloud): _description_
#         celltree (str, optional): _description_. Defaults to "default".
#         verbose (bool, optional): _description_. Defaults to True.

#     Returns:
#         _type_: _description_
#     """
#     if celltree == "default":
#         celltree = nbtrees.tree_complete_aligned()
#     data_all = None
#     predicted_nodes_all = []

#     torch.manual_seed(manual_seed_impc_transform)
#     np.random.seed(manual_seed_impc_transform)
#     for counter, sample_data in enumerate(impc):
#         torch.manual_seed(manual_seed_impc_transform)
#         np.random.seed(manual_seed_impc_transform)
#         if verbose:
#             print("\n", impc.raw_file_names[counter])
#         tmp_data = datatable.Frame(sample_data.pos.numpy())
#         tmp_data["sample_name"] = impc.raw_file_names[counter]
#         tmp_data.names = impc.metadata.feature_names + ["sample_name"]

#         if verbose:
#             print("    Sort the nodes into the celltree")
#         predicted_nodes = celltree.predict(tmp_data)
#         if verbose:
#             print("       " + re.sub(r"\n", "\n       ", predicted_nodes.__repr__()))

#         predicted_nodes_all += list(predicted_nodes)
#         if data_all is None:
#             data_all = tmp_data
#         else:
#             data_all.rbind(tmp_data)
#     if verbose:
#         print("    Set celltree.id_preds() based on predicted_nodes_all")
#     celltree.id_preds(predicted_nodes_all)

#     if verbose:
#         print("    Set celltree.data")
#     celltree.data = data_all.to_pandas()
#     return celltree
