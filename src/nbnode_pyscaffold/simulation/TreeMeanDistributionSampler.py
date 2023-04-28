import os
import pickle
from typing import Any, Dict, List, Tuple, Union

import pandas as pd

from nbnode_pyscaffold.simulation.FlowSimulationTree import FlowSimulationTreeDirichlet
from nbnode_pyscaffold.simulation.sim_target import sim_target

import numpy as np


class PseudoTorchDistributionNormal:
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

    def sample(self):
        return np.random.normal(loc=self.loc, scale=self.scale, size=1)


try:
    import torch
    import torch.distributions as D

    def mean_dist_fun(original_mean: float) -> D.Distribution:
        # You can also set that a lambda-function, e.g.
        #   lambda original_mean: D.Normal(loc=original_mean + 0, scale=1)
        return D.Normal(loc=original_mean + 0, scale=1)

except ImportError:

    def mean_dist_fun(original_mean: float) -> PseudoTorchDistributionNormal:
        """A function that returns a distribution for the new mean.

        This is a fallback function that is used if torch is not installed.
        Within TreeMeanDistributionSampler, this function is used to sample a distribution
        for the new mean. The distribution is then used to sample a new mean for
        the population that is to be changed.

        So the calls are::

            mean_distribution = mean_dist_fun(new_mean)
            new_value_from_distribution = mean_distribution.sample()

        Args:
            original_mean (float): The mean of the normal distribution

        Returns:
            Pseudo-D.Distribution:
                Mimics the torch.distributions.Distribution class in the sense that
                it has a sample() method that returns a new value from the distribution.
        """

        return PseudoTorchDistributionNormal(loc=original_mean + 0, scale=1)


class TreeMeanDistributionSampler:
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
        only_return_sampled_cell_numbers=False,
        save_changed_parameters=False,
        minimum_target_mean_proportion=1e-9,
    ) -> None:
        if isinstance(flowsim_tree, str):
            with open(
                flowsim_tree,
                "rb",
            ) as f:
                flowsim_tree: FlowSimulationTreeDirichlet = pickle.load(f)
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

        self.save_changed_parameters = save_changed_parameters
        self._only_return_sampled_cell_numbers = only_return_sampled_cell_numbers
        self.minimum_target_mean_proportion = minimum_target_mean_proportion

    @staticmethod
    def _sample(
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
        _only_return_sampled_cell_numbers=False,
        save_changed_parameters=False,
        minimum_target_mean_proportion=1e-9,
    ) -> Tuple[pd.DataFrame, Dict[str, Any], List[pd.DataFrame]]:
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
        if verbose:
            for pop, perc in flowsim_tree.mean_leafs.items():
                print(f"{perc:.6f}  {pop:125}")
        # *100 to have the distribution work on the mean proportion in percentages
        target_mean_dist = mean_distribution(original_mean * 100)
        all_true_popcounts = None
        all_changed_parameters = None
        all_sampled_samples = None
        all_targets = []
        for sample_i in range(n_samples):
            if verbose:
                print(".", end="")
            # Set a seed for the target value of the sample. (reproducibility)
            try: 
                torch.manual_seed(seed_sample_0 + sample_i + 12947)
            except NameError:
                # Then torch is not installed and the 
                pass
            np.random.seed(seed_sample_0 + sample_i + 12947)
            # raise ValueError(seed_sample_0 + sample_i + 12947)

            target_percentage = float(target_mean_dist.sample())
            all_targets += [
                max(
                    minimum_target_mean_proportion,
                    target_percentage / 100.0,
                )
            ]
            true_popcounts, changed_parameters, sampled_samples = sim_target(
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
                only_return_sampled_cell_numbers=_only_return_sampled_cell_numbers,
                save_changed_parameters=save_changed_parameters,
            )
            if all_true_popcounts is None:
                all_true_popcounts = true_popcounts
                all_changed_parameters = changed_parameters
                all_sampled_samples = sampled_samples
            else:
                all_true_popcounts = pd.concat(
                    [all_true_popcounts, true_popcounts], axis=1
                )
                all_changed_parameters += changed_parameters
                all_sampled_samples += sampled_samples
        if save_dir is not None:
            all_true_popcounts.to_csv(f"{save_dir}.csv")
        if verbose:
            print("")
            print(all_true_popcounts.apply(lambda x: x.mean(), axis=1) / n_cells)
        return (
            all_true_popcounts,
            all_changed_parameters,
            all_sampled_samples,
            all_targets,
        )

    def sample(self):
        return self._sample(
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
            _only_return_sampled_cell_numbers=self._only_return_sampled_cell_numbers,
            save_changed_parameters=self.save_changed_parameters,
            minimum_target_mean_proportion = self.minimum_target_mean_proportion
        )
