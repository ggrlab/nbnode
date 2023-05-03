from typing import Any, List, Union

import datatable
import numpy as np
import pandas as pd


def frame_cov(dt_frame: datatable.Frame) -> pd.DataFrame:
    """Compute the covariance matrix of a datatable frame from all columns.

    Similar to pd.DataFrame.cov().

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
    x: pd.DataFrame,
    fun_name: str,
    include_features: Union[List[Union[str, int]], slice] = None,
    *fun_args,
    **fun_kwargs
) -> Union[pd.DataFrame, Any]:
    """per_node_data_fun.

    To be used in NBnode.node.apply() to apply a function to the data of
    each node.


    Args:
        x (pd.DataFrame):
            A dataframe, usually the NBnode.data attribute
        include_features (Union[List[Union[str, int]], slice]):
            the given function ``fun_name`` will be applied to only these features
        fun_name (str):
            Name of the function, is usually retrieved by getattr(x, fun_name).
            Therefore, if x is e.g. an instance of datatable.Frame, fun_name can be any
            function applicable to a datatable.Frame, e.g. "mean", "sum".
            If x is e.g. an instance of pd.DataFrame, fun_name can be any function
            applicable to a pd.DataFrame, e.g. "mean", "sum", "cov".

            Special cases:

                "cov": compute the covariance matrix of the given features

    Returns:
        pd.DataFrame:
            Usually, but not necessarily a pd.DataFrame. Depends on the function.

    Examples::

        node.apply(
            lambda x: per_node_data_fun(
                x=x, include_features=include_features, fun_name="mean"
            ),
            result_attribute_name="mean",
        )
    """
    x = datatable.Frame(x)
    if include_features is not None:
        x = x[:, include_features]

    if fun_name == "cov":
        return frame_cov(x)
    else:
        res_frame = getattr(x, fun_name)(*fun_args, **fun_kwargs)
        return res_frame.to_pandas()
