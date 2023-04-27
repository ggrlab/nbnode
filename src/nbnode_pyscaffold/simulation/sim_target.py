import copy
import os
import warnings
from typing import Any, Dict, List, Tuple

import pandas as pd

from nbnode_pyscaffold.simulation.FlowSimulationTree import FlowSimulationTreeDirichlet


def sim_target(
    flowsim: FlowSimulationTreeDirichlet,
    change_pop_mean_target: List[Dict[str, float]] = [
        {"/AllCells/CD4+/CD8-/Tem": 0.05}
    ],
    n_cells=25000,
    use_only_diagonal_covmat=True,
    save_dir="sim/intraassay/sim00_target",
    sample_name=None,
    seed_sample_0=129873,
    verbose=False,
    only_return_sampled_cell_numbers=True,
    save_changed_parameters=False,
) -> Tuple[pd.DataFrame, Dict[str, Any], List[pd.DataFrame]]:
    """
        This function simulates new cells (`n_cells`) for `n_samples` samples according to the given
        flow simulation `flowsim`.

            1. flowsim.reset_populations() (for consistency)
            2. For every list element in change_pop_mean_target, change all contained populations (keys)
            to their respective mean proportions (values). Values outside (0, 1) are not allowed.

            with their respective value and changed by `flowsim.new_pop_mean(old_mean * change_prop)`
            2. Generate `n_samples` with `n_cells` are sampled from the changed FlowSimulation.
            3. (optional) The generated samples are saved to save_dir
            3. The actual number of cells and the changed parameters are returned


    Args:
        n_samples (int, optional):
            The number of simulated samples of n_cells.

            Defaults to 100.
        n_cells (int, optional):
            The number of cells per sample.

            Defaults to 25000.
        use_only_diagonal_covmat (bool, optional):
            If False, the complete covariance matrix per cell population is used to draw new cells
            If True, all off-diagonal elements of the covariance matrix are set to 0.

            Defaults to True.
        change_pop_mean_target (List[Dict[str, float]]):
            A dictionary of which cell population(s) should be changed to which mean proportion of
            all cells. floats should be between (0, 1)

        save_dir (str, optional):
            If given, the created samples (n cells X p markers) are saved into that directory as f"sample_{sample_i}.csv".

            Defaults to "sim/intraassay/sim00_target".

        seed_sample_0 (int):
            flowsim.set_seed(seed_sample_0 + sample_i)

        verbose (bool, optional):
            Verboseness. Defaults to True.

        only_return_sampled_cell_numbers (bool):
            The following variable should only be turned off if you do NOT want the cells actually
            generated, but instead want to return only the cell NUMBERS per population
    Returns:
        Tuple:
            pd.DataFrame:   Returns the true number of generated cells per leaf-population.
            Dict:           A dee copy of `flowsim.population_parameters`
    """

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    if len(change_pop_mean_target) > 1:
        warnings.warn(
            "Changing more than 1 population is NOT setting the means to the specified values - only the LAST value will be exactly the set percentage"
        )
    if isinstance(change_pop_mean_target, dict):
        # Then a single dictionary was given (and only one sample should be generated)
        change_pop_mean_target = [change_pop_mean_target]
    # Undo any changes to the populations after creating the flowsimulation
    flowsim.reset_populations()

    ncells_A_df = None
    generated_samples = []
    changed_parameters = []
    for sample_i, sample_targets in enumerate(change_pop_mean_target):
        # Change the given population means
        for pop_x, pop_target_proportion in sample_targets.items():
            if pop_target_proportion <= 0:
                raise ValueError(
                    "Changing to a target proportion of <=0 does not work properly"
                )
            if pop_target_proportion >= 1:
                raise ValueError(
                    "Changing to a target proportion of >=1 does not work properly"
                )

            flowsim.new_pop_mean(
                population_node_full_name=pop_x,
                percentage=pop_target_proportion,
            )

            if save_changed_parameters:
                changed_parameters += [copy.deepcopy(flowsim.population_parameters)]

            flowsim.set_seed(seed_sample_0 + sample_i)
            if sample_name is None:
                sample_name = f"sample_{sample_i}"

            if only_return_sampled_cell_numbers:
                ncells_A = flowsim.sample_populations(n_cells=n_cells)
            else:
                sample_A, ncells_A = flowsim.sample(
                    n_cells=n_cells,
                    return_sampled_cell_numbers=True,
                    use_only_diagonal_covmat=use_only_diagonal_covmat,
                )
                generated_samples.append(sample_A)

                if save_dir is not None:
                    current_filepath = os.path.join(save_dir, f"sample_{sample_i}.csv")
                    sample_A.to_csv(current_filepath, index=False)
                    if verbose:
                        print(f"Saved {current_filepath}")

            if ncells_A_df is None:
                ncells_A_df = pd.DataFrame(ncells_A)
                ncells_A_df.columns = [sample_name]
            else:
                ncells_A_df[sample_name] = ncells_A

            # Undo the changes of the given population means
            flowsim.reset_populations()
    return ncells_A_df, changed_parameters, generated_samples
