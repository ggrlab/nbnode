from typing import Union

from nbnode.simulation.FlowSimulationTree import FlowSimulationTreeDirichlet
from nbnode.simulation.generator_proportional import GenerateProportional


def sim00_baseline(
    flowsim_tree: Union[str, FlowSimulationTreeDirichlet],
    n_samples=100,
    n_cells=10000,
    use_only_diagonal_covmat=False,
    verbose=True,
    seed_sample_0=129873,
    save_dir="sim/sim00_pure_estimate",
    debugging_only_return_sampled_cell_numbers=False,
):
    proportional_generator = GenerateProportional(
        change_pop_mean_proportional={},
        flowsim_tree=flowsim_tree,
        n_samples=n_samples,
        n_cells=n_cells,
        use_only_diagonal_covmat=use_only_diagonal_covmat,
        verbose=verbose,
        seed_sample_0=seed_sample_0,
        save_dir=save_dir,
        debugging_only_return_sampled_cell_numbers=debugging_only_return_sampled_cell_numbers,
    )
    return proportional_generator.generate()
