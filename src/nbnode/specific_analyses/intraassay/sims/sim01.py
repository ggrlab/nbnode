from typing import Union

from nbnode.simulation.FlowSimulationTree import FlowSimulationTreeDirichlet
from nbnode.simulation.TreeMeanRelative import TreeMeanRelative


def sim01_double_tcm(
    flowsim_tree: Union[str, FlowSimulationTreeDirichlet],
    n_samples=100,
    n_cells=10000,
    use_only_diagonal_covmat=False,
    verbose=True,
    seed_sample_0=129873,
    save_dir="sim/sim01_double_tcm",
    only_return_sampled_cell_numbers=False,
):
    """Double Tcm population

    Generates a TreeMeanRelative simulation where /AllCells/CD4+/CD8-/Tcm
    proportion is doubled

    Args:
        flowsim_tree (Union[str, FlowSimulationTreeDirichlet]):
            See ``TreeMeanRelative``.
        n_samples (int, optional):
            See ``TreeMeanRelative``. Defaults to 100.
        n_cells (int, optional):
            See ``TreeMeanRelative``. Defaults to 10000.
        use_only_diagonal_covmat (bool, optional):
            See ``TreeMeanRelative``. Defaults to False.
        verbose (bool, optional):
            See ``TreeMeanRelative``. Defaults to True.
        seed_sample_0 (int, optional):
            See ``TreeMeanRelative``. Defaults to 129873.
        save_dir (str, optional):
            See ``TreeMeanRelative``.
            Defaults to "sim/sim01_double_tcm".
        only_return_sampled_cell_numbers (bool, optional):
            See ``TreeMeanRelative``.
            Defaults to False.

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any], List[pd.DataFrame]]:
            From proportional_generator.sample()
    """
    proportional_generator = TreeMeanRelative(
        change_pop_mean_proportional={
            # Changing CD4+Tems was just for testing,
            # but worked (in the sense of not changing stuff)
            # "/AllCells/CD4+/CD8-/Tem": 1,
            "/AllCells/CD4+/CD8-/Tcm": 2,
        },
        flowsim_tree=flowsim_tree,
        n_samples=n_samples,
        n_cells=n_cells,
        use_only_diagonal_covmat=use_only_diagonal_covmat,
        verbose=verbose,
        seed_sample_0=seed_sample_0,
        save_dir=save_dir,
        only_return_sampled_cell_numbers=only_return_sampled_cell_numbers,
        save_changed_parameters=True,
    )
    return proportional_generator.sample()
