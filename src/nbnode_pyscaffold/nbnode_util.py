import datatable
import numpy as np
import pandas as pd


def frame_cov(dt_frame: datatable.Frame) -> pd.DataFrame:
    """
    Compute the covariance matrix of a datatable frame from all columns similar to 
    pd.DataFrame.cov().

    Args:
        dt_frame (datatable.Frame): 
            The datatable frame to compute the covariance matrix from

    Returns:
        _type_: 
            pd.DataFrame of the covariance matrix

    """
    
    f_cols = dt_frame.export_names()
    f_cols_names = dt_frame.names
    cov_array = np.zeros((len(f_cols), len(f_cols)))
    for col1_i in range(len(f_cols)):
        for col2_i in range(col1_i, len(f_cols)):
            cov_array[col1_i, col2_i] = dt_frame[
                :, [datatable.cov(f_cols[col1_i], f_cols[col2_i])]
            ].to_numpy()[0, 0]

    # double np.diag to create a matrix from the diagonal again.
    cov_array = cov_array + cov_array.T - np.diag(np.diag(cov_array))
    return pd.DataFrame(cov_array, columns=f_cols_names, index=f_cols_names)


def per_node_data_fun(
    x: pd.DataFrame, include_features, fun_name, *fun_args, **fun_kwargs
):
    x = datatable.Frame(x)[:, include_features]
    if fun_name == "cov":
        return frame_cov(x)
    else:
        res_frame = getattr(x, fun_name)(*fun_args, **fun_kwargs)
        return res_frame.to_pandas()
