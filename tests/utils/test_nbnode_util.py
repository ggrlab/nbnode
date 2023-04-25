import datatable as dt
import numpy as np
import pandas as pd

from nbnode_pyscaffold.nbnode_util import frame_cov


def test_frame_cov():
    dt_frame = dt.Frame(np.random.rand(10, 5))
    pd_covmat = pd.DataFrame(dt_frame.to_numpy()).cov()
    dt_covmat = frame_cov(dt_frame)
    assert np.allclose(pd_covmat.to_numpy(), dt_covmat.to_numpy())
