import copy
import os
import warnings
from typing import Any, Dict, List, Tuple

import pandas as pd

from nbnode.simulation.FlowSimulationTree import FlowSimulationTreeDirichlet


def sim_proportional(
    flowsim: FlowSimulationTreeDirichlet,
    n_samples=100,
    n_cells=25000,
    use_only_diagonal_covmat=True,
    change_pop_mean_proportional={"/AllCells/CD4+/CD8-/Tem": 1},
    save_dir="sim/intraassay/sim00_baseline",
    seed_sample_0=129873,
    verbose=False,
    only_return_sampled_cell_numbers=False,
) -> Tuple[pd.DataFrame, Dict[str, Any], List[pd.DataFrame]]:
    """
        This function simulates new cells (`n_cells`) for `n_samples` samples
        according to the given flow simulation `flowsim`.

            1. The population mean of the keys from `change_pop_mean_proportional` are
            multiplied with their respective value and changed by
            `flowsim.new_pop_mean(old_mean * change_prop)`
            2. Generate `n_samples` with `n_cells` are sampled from the changed
            FlowSimulation.
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
            If False, the complete covariance matrix per cell population is used to
            draw new cells
            If True, all off-diagonal elements of the covariance matrix are set to 0.

            Defaults to True.
        change_pop_mean_proportional (dict, optional):
            A dictionary of which cell population(s) should be changed by which
            fraction.
            A value of 1 does not change the mean proportion of the cell population.
            The changes to `flowsim` are not persistent as they are undone after the
            simulation.

            Defaults to {"/AllCells/CD4+/CD8-/Tem": 1}.
        save_dir (str, optional):
            If given, the created samples (n cells X p markers) are saved into that
            directory as f"sample_{sample_i}.csv".

            Defaults to "sim/intraassay/sim00_baseline".

        seed_sample_0 (int):
            flowsim.set_seed(seed_sample_0 + sample_i)
        verbose (bool, optional):
            Verboseness. Defaults to True.

        only_return_sampled_cell_numbers (bool):
            If true, only the number of cells per population are returned,
            not the actual samples.
    Returns:
        Tuple:
            pd.DataFrame:
                Returns the true number of generated cells per leaf-population.
            Dict:
                A deep copy of `flowsim.population_parameters`
    """

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    if len(change_pop_mean_proportional) > 1:
        warnings.warn(
            "Changing more than 1 population is NOT setting the means to the "
            + "specified values - only the LAST value will be "
            + "exactly the set percentage"
        )
    # Change the given population means
    for pop_x, change_prop in change_pop_mean_proportional.items():
        if verbose:
            print(
                pop_x,
                "  ",
                flowsim.pop_mean(pop_x),
                " -> ",
                flowsim.pop_mean(pop_x) * change_prop,
            )
        flowsim.new_pop_mean(
            population_node_full_name=pop_x,
            percentage=flowsim.pop_mean(pop_x) * change_prop,
        )
    if verbose:
        for pop, perc in flowsim.mean_leafs.items():
            print(f"{perc:.6f}  {pop:125}")

    # Actually sample
    ncells_A_df = None
    generated_samples = []
    for sample_i in range(n_samples):
        flowsim.set_seed(seed_sample_0 + sample_i)
        sample_name = f"sample_{sample_i}"
        # print(sample_name)
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

    changed_parameters = copy.deepcopy(flowsim.population_parameters)
    # Undo the changes of the given population means
    for pop_x, change_prop in change_pop_mean_proportional.items():
        if verbose:
            print(
                pop_x,
                "  ",
                flowsim.pop_mean(pop_x),
                " -> ",
                flowsim.pop_mean(pop_x) / change_prop,
            )
        flowsim.new_pop_mean(
            population_node_full_name=pop_x,
            percentage=flowsim.pop_mean(pop_x) / change_prop,
        )
    return ncells_A_df, changed_parameters, generated_samples
