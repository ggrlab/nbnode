from typing import Union

from nbnode.simulation.FlowSimulationTree import FlowSimulationTreeDirichlet
from nbnode.simulation.TreeMeanRelative import TreeMeanRelative


def sim02_temra(
    flowsim_tree: Union[str, FlowSimulationTreeDirichlet],
    n_samples=100,
    n_cells=10000,
    use_only_diagonal_covmat=False,
    verbose=True,
    seed_sample_0=129873,
    save_dir="sim/sim02_temra",
    only_return_sampled_cell_numbers=False,
):
    """Change /AllCells/CD4-/CD8+/Temra proportion from 7.17% to 33.23%

    Generates a TreeMeanRelative simulation where /AllCells/CD4-/CD8+/Temra
    proportion is increased from 7.17% (baseline) to 33.23% (target).

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
            Defaults to "sim/sim02_temra".
        only_return_sampled_cell_numbers (bool, optional):
            See ``TreeMeanRelative``.
            Defaults to False.

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any], List[pd.DataFrame]]:
            From proportional_generator.sample()
    """
    proportional_generator = TreeMeanRelative(
        change_pop_mean_proportional={
            # 7.17% --> 33.23%
            "/AllCells/CD4-/CD8+/Temra": 4.634588563458856,
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
