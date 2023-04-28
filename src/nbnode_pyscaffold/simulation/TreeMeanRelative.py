import os
import pickle
from typing import Any, Dict, List, Tuple, Union

import pandas as pd

from nbnode_pyscaffold.simulation.FlowSimulationTree import FlowSimulationTreeDirichlet
from nbnode_pyscaffold.simulation.sim_proportional import sim_proportional


class TreeMeanRelative:
    def __init__(
        self,
        flowsim_tree: Union[str, FlowSimulationTreeDirichlet],
        change_pop_mean_proportional: Dict[str, float],
        n_samples=100,
        n_cells=10000,
        use_only_diagonal_covmat=False,
        verbose=True,
        seed_sample_0=129873,
        save_dir="sim/sim00_pure_estimate",
        only_return_sampled_cell_numbers=False,
    ) -> None:
        self.flowsim_tree = flowsim_tree
        self.change_pop_mean_proportional = change_pop_mean_proportional
        self.n_samples = n_samples
        self.n_cells = n_cells
        self.use_only_diagonal_covmat = use_only_diagonal_covmat
        self.verbose = verbose
        self.seed_sample_0 = seed_sample_0
        self.save_dir = save_dir
        # The following variable should only be turned off if you do NOT want the cells actually
        # sampled, but instead want to return only the cell NUMBERS per population
        self._only_return_sampled_cell_numbers = (
            only_return_sampled_cell_numbers
        )

    @staticmethod
    def _sample(
        flowsim_tree: Union[str, FlowSimulationTreeDirichlet],
        n_samples=100,
        n_cells=10000,
        change_pop_mean_proportional={},
        use_only_diagonal_covmat=False,
        verbose=True,
        seed_sample_0=129873,
        save_dir="sim/sim00_pure_estimate",
        _only_return_sampled_cell_numbers=False,
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
        true_popcounts, changed_parameters, sampled_samples = sim_proportional(
            flowsim=flowsim_tree,
            n_samples=n_samples,
            n_cells=n_cells,
            use_only_diagonal_covmat=use_only_diagonal_covmat,
            change_pop_mean_proportional=change_pop_mean_proportional,
            seed_sample_0=seed_sample_0,
            save_dir=save_dir,
            only_return_sampled_cell_numbers=_only_return_sampled_cell_numbers,
        )
        if save_dir is not None:
            true_popcounts.to_csv(f"{save_dir}.csv")
            changed_parameters["alpha"].to_csv(f"{save_dir}_parameters.csv")
        if verbose:
            print(true_popcounts.apply(lambda x: x.mean(), axis=1) / n_cells)
        return true_popcounts, changed_parameters, sampled_samples

    def sample(self):
        return self._sample(
            flowsim_tree=self.flowsim_tree,
            n_samples=self.n_samples,
            n_cells=self.n_cells,
            change_pop_mean_proportional=self.change_pop_mean_proportional,
            use_only_diagonal_covmat=self.use_only_diagonal_covmat,
            verbose=self.verbose,
            seed_sample_0=self.seed_sample_0,
            save_dir=self.save_dir,
            _only_return_sampled_cell_numbers=self._only_return_sampled_cell_numbers,
        )

    def sample_customize(
        self,
        n_samples=None,
        n_cells=None,
        change_pop_mean_proportional=None,
        use_only_diagonal_covmat=None,
        verbose=None,
        seed_sample_0=None,
        save_dir=None,
        _only_return_sampled_cell_numbers=None,
    ):
        return self._sample(
            n_samples=self.n_samples if n_samples is None else n_samples,
            n_cells=self.n_cells if n_cells is None else n_cells,
            change_pop_mean_proportional=self.change_pop_mean_proportional
            if change_pop_mean_proportional is None
            else change_pop_mean_proportional,
            use_only_diagonal_covmat=self.use_only_diagonal_covmat
            if use_only_diagonal_covmat is None
            else use_only_diagonal_covmat,
            verbose=self.verbose if verbose is None else verbose,
            seed_sample_0=self.seed_sample_0
            if seed_sample_0 is None
            else seed_sample_0,
            save_dir=self.save_dir if save_dir is None else save_dir,
            _only_return_sampled_cell_numbers=self._only_return_sampled_cell_numbers
            if _only_return_sampled_cell_numbers is None
            else _only_return_sampled_cell_numbers,
        )
