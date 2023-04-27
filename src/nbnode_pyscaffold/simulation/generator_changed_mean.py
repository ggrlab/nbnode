import os
import pickle
from typing import Any, Dict, List, Tuple, Union

import pandas as pd
import torch.distributions as D

from nbnode_pyscaffold.simulation.FlowSimulationTree import FlowSimulationTreeDirichlet
from nbnode_pyscaffold.simulation.sim_target import sim_target


def mean_dist_fun(original_mean: float) -> D.Distribution:
    # You can also set that a lambda-function, e.g.
    #   lambda original_mean: D.Normal(loc=original_mean + 0, scale=1)
    return D.Normal(loc=original_mean + 0, scale=1)


class GenerateChangedMean:
    def __init__(
        self,
        flowsim_tree: Union[str, FlowSimulationTreeDirichlet],
        population_name_to_change: str,
        mean_distribution=mean_dist_fun,
        n_samples=100,
        n_cells=10000,
        use_only_diagonal_covmat=False,
        verbose=True,
        seed_sample_0=129873,
        save_dir="sim/sim00_m0.sd1",
        debugging_only_return_sampled_cell_numbers=True,
    ) -> None:
        self.flowsim_tree = flowsim_tree
        self.n_samples = n_samples
        self.n_cells = n_cells
        self.use_only_diagonal_covmat = use_only_diagonal_covmat
        self.verbose = verbose
        self.seed_sample_0 = seed_sample_0
        self.save_dir = save_dir

        self.population_name_to_change = population_name_to_change
        self.original_mean = flowsim_tree.pop_mean(
            population_node_full_name=population_name_to_change
        )
        self.mean_distribution = mean_distribution

        # The following variable should only be turned off if you do NOT want the cells actually
        # generated, but instead want to return only the cell NUMBERS per population
        self._debugging_only_return_sampled_cell_numbers = (
            debugging_only_return_sampled_cell_numbers
        )

    @staticmethod
    def _generate(
        flowsim_tree: Union[str, FlowSimulationTreeDirichlet],
        population_name_to_change: float,
        original_mean: float,
        mean_distribution,
        n_samples=100,
        n_cells=10000,
        use_only_diagonal_covmat=False,
        verbose=True,
        seed_sample_0=129873,
        save_dir="sim/sim00_pure_estimate",
        _debugging_only_return_sampled_cell_numbers=True,
    ) -> Tuple[pd.DataFrame, Dict[str, Any], List[pd.DataFrame]]:
        if isinstance(flowsim_tree, str):
            with open(
                flowsim_tree,
                "rb",
            ) as f:
                flowsim_tree: FlowSimulationTreeDirichlet = pickle.load(f)

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
        if verbose:
            for pop, perc in flowsim_tree.mean_leafs.items():
                print(f"{perc:.6f}  {pop:125}")
        # *100 to have the distribution work on the mean proportion in percentages
        target_mean_dist: D.Distribution = mean_distribution(original_mean * 100)
        all_true_popcounts = None
        all_changed_parameters = None
        all_generated_samples = None
        all_targets = []
        for sample_i in range(n_samples):
            if verbose:
                print(".", end="")
            all_targets += [max(1e-9, float(target_mean_dist.sample()) / 100.0)]
            true_popcounts, changed_parameters, generated_samples = sim_target(
                flowsim=flowsim_tree,
                change_pop_mean_target={
                    # Use the max to omit negative and 0 values
                    population_name_to_change: all_targets[-1],
                },
                n_cells=n_cells,
                use_only_diagonal_covmat=use_only_diagonal_covmat,
                seed_sample_0=seed_sample_0 + sample_i,
                save_dir=save_dir,
                sample_name=f"sample_{sample_i}",
                only_return_sampled_cell_numbers=_debugging_only_return_sampled_cell_numbers,
                save_changed_parameters = False
            )
            if all_true_popcounts is None:
                all_true_popcounts = true_popcounts
                all_changed_parameters = changed_parameters
                all_generated_samples = generated_samples
            else:
                all_true_popcounts = pd.concat([all_true_popcounts, true_popcounts], axis=1)
                all_changed_parameters += changed_parameters
                all_generated_samples += generated_samples
        if save_dir is not None:
            all_true_popcounts.to_csv(f"{save_dir}.csv")
        if verbose:
            print("")
            print(all_true_popcounts.apply(lambda x: x.mean(), axis=1) / n_cells)
        return all_true_popcounts, all_changed_parameters, all_generated_samples, all_targets

    def generate(self):
        return self._generate(
            flowsim_tree=self.flowsim_tree,
            n_samples=self.n_samples,
            n_cells=self.n_cells,
            population_name_to_change=self.population_name_to_change,
            original_mean=self.original_mean,
            mean_distribution=self.mean_distribution,
            # change_pop_mean_proportional=self.change_pop_mean_proportional,
            use_only_diagonal_covmat=self.use_only_diagonal_covmat,
            verbose=self.verbose,
            seed_sample_0=self.seed_sample_0,
            save_dir=self.save_dir,
            _debugging_only_return_sampled_cell_numbers=self._debugging_only_return_sampled_cell_numbers,
        )
