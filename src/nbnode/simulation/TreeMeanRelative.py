import os
import pickle
from typing import Any, Dict, List, Tuple, Union

import pandas as pd

from nbnode.simulation.FlowSimulationTree import FlowSimulationTreeDirichlet
from nbnode.simulation.sim_proportional import sim_proportional


class TreeMeanRelative:
    """Sample from a tree with a relative change in a population."""

    def __init__(
        self,
        flowsim_tree: Union[str, FlowSimulationTreeDirichlet],
        change_pop_mean_proportional: Dict[str, float],
        n_samples=100,
        n_cells=10000,
        use_only_diagonal_covmat=False,
        verbose=False,
        seed_sample_0=129873,
        save_dir="sim/sim00_pure_estimate",
        only_return_sampled_cell_numbers=False,
        save_changed_parameters=True,
    ) -> None:
        """Sample from a tree with a relative change in a population mean.

            TreeMeanRelative changes the concentration of a cell population by a
            certain factor (e.g. old = 173.2, new = old*2) and all other concentration
            parameters decrease (by a factor) such that the overall sum of
            concentration parameters remains equal.

        Args:
            flowsim_tree (Union[str, FlowSimulationTreeDirichlet]):
                A FlowSimulationTreeDirichlet object or a path to a pickle file.
            change_pop_mean_proportional (Dict[str, float]):
                The name (keys) of the population(s) that are to be changed. The
                values are the proportional change in the population mean.
                E.g. ::

                    {
                        "/AllCells/DP": 2.0,
                    }

                This will change the mean of the population "/AllCells/DP" by a factor
                of 2.0. All other populations will be changed such that the overall sum
                of concentration parameters remains equal.

            n_samples (int, optional):
                The number of samples to draw from the tree. Defaults to 100.
            n_cells (int, optional):
                The number of cells per sample. Defaults to 10000.
            use_only_diagonal_covmat (bool, optional):
                Whether to use only the diagonal of the covariance matrix.
                Defaults to False.
            verbose (bool, optional):
                Verbosity. Defaults to False.
            seed_sample_0 (int, optional):
                The seed for the first sample. The seed for the i-th sample is
                seed_sample_0 + i.
                Defaults to 129873.
            save_dir (str, optional):
                If not None, the synthesized samples  are saved to this directory.
                Defaults to "sim/sim00_pure_estimate".
            only_return_sampled_cell_numbers (bool, optional):
                Whether to only return the sampled cell numbers.
                Defaults to False.
            save_changed_parameters (bool, optional):
                Whether to save+return the changed parameters. Defaults to True.

        """
        if isinstance(flowsim_tree, str):
            with open(
                flowsim_tree,
                "rb",
            ) as f:
                flowsim_tree: FlowSimulationTreeDirichlet = pickle.load(f)
        self.flowsim_tree = flowsim_tree

        self.change_pop_mean_proportional = change_pop_mean_proportional
        self.n_samples = n_samples
        self.n_cells = n_cells
        self.use_only_diagonal_covmat = use_only_diagonal_covmat
        self.verbose = verbose
        self.seed_sample_0 = seed_sample_0
        self.save_dir = save_dir
        self.save_changed_parameters = save_changed_parameters
        self._only_return_sampled_cell_numbers = only_return_sampled_cell_numbers

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
        save_changed_parameters=False,
    ) -> Tuple[pd.DataFrame, Dict[str, Any], List[pd.DataFrame]]:
        """A static method to sample with a relative change in a population mean.

        See the __init__ method for the description of the arguments.

        Returns:
            Tuple[pd.DataFrame, Dict[str, Any], List[pd.DataFrame]]:
                - A dataframe with the sampled cell numbers.
                - A dictionary with the parameters of the
                  dirichlet distribution.
                - A list of dataframes with the sampled cell matrices
                  (n_cells X features) for each sample.
        """
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
        if not save_changed_parameters:
            changed_parameters = None
        if save_dir is not None:
            true_popcounts.to_csv(f"{save_dir}.csv")
            if save_changed_parameters:
                changed_parameters["alpha"].to_csv(f"{save_dir}_parameters.csv")
        if verbose:
            print(true_popcounts.apply(lambda x: x.mean(), axis=1) / n_cells)
        return true_popcounts, changed_parameters, sampled_samples

    def sample(self) -> Tuple[pd.DataFrame, Dict[str, Any], List[pd.DataFrame]]:
        """A method to sample with a relative change in a population mean.

        See the __init__ method for the description of the arguments.

        Returns:
            Tuple[pd.DataFrame, Dict[str, Any], List[pd.DataFrame]]:
                - A dataframe with the sampled cell numbers.
                - A dictionary with the parameters of the
                  dirichlet distribution.
                - A list of dataframes with the sampled cell matrices
                  (n_cells X features) for each sample.
        """
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
            save_changed_parameters=self.save_changed_parameters,
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
        save_changed_parameters=False,
    ) -> Tuple[pd.DataFrame, Dict[str, Any], List[pd.DataFrame]]:
        """A customizable method to sample with a relative change in a population mean.

        See the __init__ method for the description of the arguments.
        In contrast to ``.sample()``, this method allows to change each parameter
        individually.
        If any argument is not given, the default value set in __init__ method
        will be used.

        Returns:
            Tuple[pd.DataFrame, Dict[str, Any], List[pd.DataFrame]]:
                - A dataframe with the sampled cell numbers.
                - A dictionary with the parameters of the
                  dirichlet distribution.
                - A list of dataframes with the sampled cell matrices
                  (n_cells X features) for each sample.
        """
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
            save_changed_parameters=self.save_changed_parameters
            if save_changed_parameters is None
            else save_changed_parameters,
        )
