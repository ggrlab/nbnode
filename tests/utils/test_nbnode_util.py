import os

import datatable as dt
import numpy as np
import pandas as pd

from nbnode.nbnode_util import frame_cov, per_node_data_fun
from nbnode.testutil.helpers import find_dirname_above_currentfile

TESTS_DIR = find_dirname_above_currentfile()


def test_frame_cov():
    dt_frame = dt.Frame(np.random.rand(10, 5))
    pd_covmat = pd.DataFrame(dt_frame.to_numpy()).cov()
    dt_covmat = frame_cov(dt_frame)
    assert np.allclose(pd_covmat.to_numpy(), dt_covmat.to_numpy())


def test_per_node_data_fun():
    import math
    import re

    import nbnode.nbnode_trees as nbtree

    cellmat = pd.read_csv(
        os.path.join(
            TESTS_DIR, "testdata", "flowcytometry", "gated_cells", "cellmat.csv"
        )
    )
    cellmat.columns = [re.sub("_.*", "", x) for x in cellmat.columns]
    celltree_trunk = nbtree.tree_complete_aligned_trunk()
    celltree_trunk.data = cellmat
    assert "mean" not in celltree_trunk.__dict__.keys()
    celltree_trunk.apply(
        lambda x: per_node_data_fun(x=x, fun_name="mean"),
        result_attribute_name="mean",
    )
    # The previous command works, but the ids are not set, therefore each fun_name
    # has no data to compute from.
    assert all([math.isnan(x) for x in celltree_trunk.mean.iloc[0, :]])

    a = celltree_trunk.predict(cellmat)
    celltree_trunk.id_preds(a)

    celltree_trunk.apply(
        lambda x: per_node_data_fun(x=x, fun_name="mean"),
        result_attribute_name="mean",
    )
    assert all([not math.isnan(x) for x in celltree_trunk.mean.iloc[0, :]])
    assert celltree_trunk.mean.shape == (1, 13)

    celltree_trunk.apply(
        lambda x: per_node_data_fun(x=x, fun_name="mean", include_features=slice(0, 3)),
        result_attribute_name="mean",
    )
    assert celltree_trunk.mean.shape == (1, 3)
    assert all(celltree_trunk.mean.columns == ["FS", "FS.0", "SS"])

    celltree_trunk.apply(
        lambda x: per_node_data_fun(
            x=x, fun_name="mean", include_features=["FS", "FS.0", "SS"]
        ),
        result_attribute_name="mean",
    )
    assert celltree_trunk.mean.shape == (1, 3)
    assert all(celltree_trunk.mean.columns == ["FS", "FS.0", "SS"])

    celltree_trunk.apply(
        lambda x: per_node_data_fun(x=x, fun_name="mean", include_features=[0, 1, 2]),
        result_attribute_name="mean",
    )
    assert celltree_trunk.mean.shape == (1, 3)
    assert all(celltree_trunk.mean.columns == ["FS", "FS.0", "SS"])

    celltree_trunk.apply(
        lambda x: per_node_data_fun(x=x, fun_name="mean", include_features=1),
        result_attribute_name="mean",
    )
    assert all(celltree_trunk.mean.columns == ["FS.0"])


def test_per_node_data_fun_covariance_matrix():
    import re

    import nbnode.nbnode_trees as nbtree

    cellmat = pd.read_csv(
        os.path.join(
            TESTS_DIR, "testdata", "flowcytometry", "gated_cells", "cellmat.csv"
        )
    )
    cellmat.columns = [re.sub("_.*", "", x) for x in cellmat.columns]
    celltree_trunk = nbtree.tree_complete_aligned_trunk()
    celltree_trunk.data = cellmat
    a = celltree_trunk.predict(cellmat)
    celltree_trunk.id_preds(a)
    celltree_trunk.apply(
        lambda x: per_node_data_fun(x=x, fun_name="cov"),
        result_attribute_name="cov",
    )
