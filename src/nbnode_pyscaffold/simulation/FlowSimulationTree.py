import copy
import warnings
from abc import abstractmethod
from importlib.metadata import PackageNotFoundError
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import anytree
import numpy as np
import pandas as pd

from nbnode_pyscaffold.nbnode import NBNode
from nbnode_pyscaffold.nbnode_util import per_node_data_fun

dirichlet_installed = True
try:
    import dirichlet
except ImportError:
    dirichlet_installed = False


class BaseFlowSimulationTree:
    def __init__(
        self,
        rootnode: NBNode,
        data_cellgroup_col: str = "sample",
        node_percentages: Optional[pd.DataFrame] = None,
        seed: int = 12987,
        include_features="dataset_melanoma",
        verbose: bool = False,
    ) -> None:
        """Base flow simulation.

        Args:
            rootnode (NBNode):
                A NBnode from which the simulation should be initiated.

            data_cellgroup_col (str, optional):
                The column of rootnode.data containing identifiers
                per cell which sample it refers to.
                If "None" and node_percentages=None, all cells are assumed to come from
                a single sample.

                Defaults to "sample".
            node_percentages (Optional[pd.DataFrame], optional):
                If node_percentages is not given, node.data MUST contain a column which
                identifies groups of cells (default: "sample")
                Defaults to None.
            seed (int, optional):
                Relevant if sampling from the resulting Simulation.
                Defaults to 12987.
            include_features (str, optional): _description_. Defaults to "dataset_melanoma".


        Attributes:
            _orig_nodes_to_leafnodes:
                pd.DataFrame from Dict[leaf_node.get_name_full(): node.get_name_full()].
                Potentially multiple `leaf_nodes` have the same `node`. If `node` is already
                a leaf node, `leaf_node = node`.
                The index of the dataframe is `leaf_node.get_name_full()`.
        """

        self.rootnode_structure = rootnode.copy_structure()
        leaf_nodes_data = [
            node for node in anytree.PreOrderIter(rootnode) if node.is_leaf
        ]
        # Sort the leaf nodes such that self.sample() gets the same sample regardless of the order
        leaf_nodes_data = sorted(leaf_nodes_data, key=lambda node: node.get_name_full())

        if len(leaf_nodes_data) == 1:
            warnings.warn(
                "Only one single leaf node found, all cells will be simulated from"
                + " single node, are you sure that is what you want?\n"
                + "The dirichlet parameter will be 1, only the estimated"
                + "cell_distributions might make sense."
            )

        if rootnode.data is None:
            raise ValueError(
                "rootnode.data is None. Please set rootnode.data before "
                + "creating the simulation. Usually: \n"
                + "    "
                + "rootnode.data = pd.DF_with_colnames_matching_the_include_features"
            )
        if len(rootnode.ids) <= 1:
            raise ValueError(
                "rootnode.ids does not contain any id.\n"
                + "In this simulation we assume that all cells originate from the root"
                + " node, therefore rootnode.ids must contain at least one id.\n"
                + "Please set rootnode.ids before creating the simulation. Usually: \n"
                + "\n    predicted_nodes_per_cell = celltree.predict(cellmat)\n"
                + "    celltree.id_preds(predicted_nodes_per_cell)\n"
            )
        #### 1. Get the percentage of cells in each LEAF-node per sample
        if node_percentages is None:
            # If node_percentages is not given, node.data MUST contain a column which identifies
            # groups of cells (default: "sample")
            if data_cellgroup_col is not None:
                ncells_per_sample = rootnode.data[data_cellgroup_col].value_counts()
                ncells_per_node_per_sample = [
                    x.data[data_cellgroup_col].value_counts() for x in leaf_nodes_data
                ]

            else:
                ncells_per_sample = rootnode.data.shape[0]
                ncells_per_node_per_sample = pd.DataFrame(
                    {
                        "single_sample": {
                            x.get_name_full(): len(x.data) for x in leaf_nodes_data
                        }
                    }
                )

            ncells_pernode_persample = pd.DataFrame(
                ncells_per_node_per_sample,
                index=[node.get_name_full() for node in leaf_nodes_data],
            )
            # n_leafnode_persample contains NaN: They are missing because there were no cells inside.
            ncells_pernode_persample = ncells_pernode_persample.fillna(0)
            node_percentages = ncells_pernode_persample.div(ncells_per_sample, axis=1)
        self.node_percentages = node_percentages

        #### 2. Estimate distribution of populations
        # Use the identified or given node_percentages
        # to calculate/estimate the distribution of how many cells are in each LEAF-node
        # Results in a dictionary of
        #   {
        #       "__name": List[str],  # population names
        #       "distribution_parameter_1": List,   # First     distribution parameter, e.g. "alpha" or "mean"
        #       "distribution_parameter_2": List,   # Second    distribution parameter, e.g. "sigma"
        #       # ...                               # Further   distribution parameters
        #   }
        # The length of each list must be identical.
        # The distribution parameters are usually pd.DataFrames
        self.population_parameters = self.estimate_population_distribution(
            node_percentages
        )
        if verbose:
            print(self.population_parameters)

        # Extension:
        # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4742377/
        # https://mail.python.org/pipermail/scipy-user/2011-August/030312.html
        # https://cmdlinetips.com/2019/05/empirical-cumulative-distribution-function-ecdf-in-python/
        # ! https://stats.stackexchange.com/questions/226935/algorithms-for-computing-multivariate-empirical-distribution-function-ecdf
        # ! https://www.sciencedirect.com/science/article/pii/S0167947321001018

        #### 3. Estimate CELL distributions per LEAF-node using `node.data`
        # The distribution is estimated only for `node.data[include_features]`
        if include_features == "dataset_melanoma":
            include_features = [
                "FS_INT",
                "FS_TOF",
                "SS_INT",
                "CD45RA_FITC",
                "CCR7_PE",
                "CD28_ECD",
                "PD1_PC5.5",
                "CD27_PC7",
                "CD4_APC",
                "CD8_APC-A700",
                "CD3_APC-A750",
                "CD57_PacBlue",
                "CD45_KrOrange",
            ]
        elif include_features == "dataset_melanoma_short":
            include_features = [
                "FS",
                "FS.0",
                "SS",
                "CD45RA",
                "CCR7",
                "CD28",
                "PD1",
                "CD27",
                "CD4",
                "CD8",
                "CD3",
                "CD57",
                "CD45",
            ]

        self.include_features = include_features
        self.nodes_info_dict = self.estimate_cell_distributions(nodes=leaf_nodes_data)

        # reset population parameters AFTER cell distribution estimation
        # because populations where the distributions could not be estimated have been removed
        self.__reset_pop_params = copy.deepcopy(self.population_parameters)
        #### 4. Set the initial seed such that .sample() gives consistent results
        self.set_seed(seed)

    def estimate_cell_distributions(self, nodes):
        include_features = self.include_features
        # Calculate mean and covariance for each of the given nodes.
        nodes_info_dict = {}
        for node in nodes:
            if len(node.ids) <= 1:
                warnings.warn(
                    f"No cells in {node.get_name_full()} to calculate any moments. Removing the population also from population_means and population_cov.",
                )
                self.remove_population(node.get_name_full())

                # If there is no cell no measures can be calculated
                # If there is 1 cell, the covariance cannot be calculated
                nodes_info_dict.update({node.get_name_full(): None})

            try:
                mu = node.mean[include_features].iloc[0]
            except AttributeError:
                node.apply(
                    lambda x: per_node_data_fun(
                        x=x, include_features=include_features, fun_name="mean"
                    ),
                    result_attribute_name="mean",
                )
                mu = node.mean[include_features].iloc[0]

            try:
                cov = node.cov.loc[include_features, include_features]
            except AttributeError:
                node.apply(
                    lambda x: per_node_data_fun(
                        x=x, include_features=include_features, fun_name="cov"
                    ),
                    result_attribute_name="cov",
                )
                cov = node.cov.loc[include_features, include_features]

            nodes_info_dict.update({node.get_name_full(): {"mu": mu, "cov": cov}})
        return nodes_info_dict

    @staticmethod
    @abstractmethod
    def estimate_population_distribution(
        node_percentages,
    ) -> Dict[Union[Literal["__name"], str], Any]:
        """_summary_

        Args:
            node_percentages (_type_): _description_

        Returns:

            Dict[str, Any]:
                {
                    # __name must be given
                    "__name": list(population_means.index),
                    "mean": population_means,   # distribution parameter
                    "cov": population_cov,      # distribution parameter
                }
        Example:

            # Calculate mean and covariance for each of the populations (rows of node_percentages)

            population_means = node_percentages.mean(axis=1)
            if node_percentages.shape[1] > 1:
                population_cov = node_percentages.T.cov()
            else:
                population_cov = np.identity(len(population_means))
                population_cov = pd.DataFrame(
                    population_cov,
                    columns=population_means.index,
                    index=population_means.index,
                )
            return {
                "__name": list(population_means.index),
                "mean": population_means,
                "cov": population_cov,
            }
        """

    @abstractmethod
    def remove_population(self, population_name: str):
        """
        Remove a certain population from the simulation. Necessary if any population had no cells
        and therefore the cell-parameters for the population cannot be estimated.
        ALWAYS call
            `self.population_parameters["__name"].remove(population_name)`

        Args:
            population_name (str):
                The name of the population which should be removed from the population_parameters.
        Example:
            self.population_parameters["__name"].remove(population_name)
            self.population_parameters["mean"].drop(
                population_name, inplace=True
            )
            self.population_parameters["cov"].drop(
                population_name, inplace=True, axis=0
            )
            self.population_parameters["cov"].drop(
                population_name, inplace=True, axis=1
            )
        """

    def reset_populations(self):
        self.population_parameters = copy.deepcopy(self.__reset_pop_params)

    def set_seed(self, seed: int):
        self._rng = np.random.default_rng(seed)

    def ncells_from_percentages(self, percentages, n_cells) -> List[int]:
        ### "Sample" the number of cells according to the random percentages
        onesample_ncells_perpop = percentages * n_cells
        onesample_ncells_perpop = np.floor(onesample_ncells_perpop)

        if sum(onesample_ncells_perpop) < n_cells:
            # because of floor there are too little cells sampled
            remaining_cells = self._rng.choice(
                [population_i for population_i in range(len(percentages))],
                size=int(n_cells - sum(onesample_ncells_perpop)),
                replace=True,
                p=percentages,
            )
            for cell_from_pop_i in remaining_cells:
                onesample_ncells_perpop[cell_from_pop_i] += 1
        return onesample_ncells_perpop

    @abstractmethod
    def generate_populations(
        self,
        population_parameters: Dict[str, Any],
        n_cells: int,
        *args,
        **kwargs,
    ) -> List[float]:
        """
        Generate a list of percentages per cell population.

        Args:
            population_parameters (Dict[str, Any]):
                Parameters for the distribution to draw from.
            n_cells (int):
                How many cells should be drawn per sample

        Returns:
            List[int]: List of percentages per cell population
        Example:

            ### 1. Generate random percentages for each population
            random_mean = self._rng.multivariate_normal(
                **population_parameters
            )
            # HACK: therefore all values are positive, this is a kindoff hack and should be replaced with a better distribution
            random_mean -= min(random_mean)
            # add the smallest value such that the "most negative population" has atleast _some_ chance of occuring
            random_mean += sorted(set(random_mean))[1] / 1e3
            # normalize to 1
            random_mean = random_mean / sum(random_mean)

            ### 2. "Sample" the number of cells according to the random percentages
            onesample_ncells_perpop = random_mean * n_cells
            onesample_ncells_perpop = np.floor(onesample_ncells_perpop)

            if sum(onesample_ncells_perpop) < n_cells:
                # because of floor there are too little cells sampled
                remaining_cells = self._rng.choice(
                    [population_i for population_i in range(len(random_mean))],
                    size=int(n_cells - sum(onesample_ncells_perpop)),
                    replace=True,
                    p=random_mean,
                )
                for cell_from_pop_i in remaining_cells:
                    onesample_ncells_perpop[cell_from_pop_i] += 1
            return onesample_ncells_perpop
        """

    def sample_populations(
        self,
        n_cells: int = 10000,
        **population_parameters,
    ) -> pd.Series:
        #### Generate number of cells according to leaf node population distributions
        if len(population_parameters) == 0:
            population_parameters = self.population_parameters

        onesample_ncells_perpop = self.generate_populations(
            population_parameters={
                key: value
                for key, value in population_parameters.items()
                if key != "__name"
            },
            n_cells=n_cells,
        )
        tmp_allpops = pd.DataFrame(
            data=onesample_ncells_perpop, index=population_parameters["__name"]
        )

        return tmp_allpops[0]  # then it becomes a pd.Series, no pd.DataFrame

    def sample(
        self,
        n_cells: int = 10000,
        return_sampled_cell_numbers: bool = False,
        use_only_diagonal_covmat: bool = True,
        **population_parameters,
    ) -> Union[Tuple[pd.DataFrame, pd.Series], pd.DataFrame]:
        #### 1. Generate number of cells according to leaf node population distributions
        onesample_ncells_perpop_df = self.sample_populations(
            n_cells=n_cells, **population_parameters
        )

        #### 2. Actually generate the cells _per population_
        # According to the estimated CELL distributions per leaf node
        all_cells = None
        for population_name, n_in_population in onesample_ncells_perpop_df.items():
            # print(population_name, n_in_population)
            cov = self.nodes_info_dict[population_name]["cov"]
            if use_only_diagonal_covmat:
                cov = np.diag(np.diag(cov))
            simulated_cells = self._rng.multivariate_normal(
                mean=self.nodes_info_dict[population_name]["mu"],
                cov=cov,
                size=int(n_in_population),
            )
            if all_cells is None:
                all_cells = simulated_cells
            else:
                all_cells = np.concatenate([all_cells, simulated_cells])

        sampled_cells = pd.DataFrame(
            all_cells,
            # feature names (=index) of any cell-distribution mean
            columns=self.nodes_info_dict[population_name]["mu"].index,
        )

        if return_sampled_cell_numbers:
            return (
                sampled_cells,
                onesample_ncells_perpop_df,
            )
        else:
            return sampled_cells


class FlowSimulationTreeDirichlet(BaseFlowSimulationTree):
    def __init__(
        self,
        rootnode: NBNode,
        # population_nodes: Optional[Union[List[NBNode], List[str]]] = None,
        data_cellgroup_col: str = "sample",
        node_percentages: pd.DataFrame = None,
        seed: int = 12987,
        include_features="dataset_melanoma",
        verbose: bool = False,
    ) -> None:
        if not dirichlet_installed:
            raise PackageNotFoundError(
                "Python package <dirichlet> has not been found, cannot simulate FlowSimulationDirichlet"
            )
        super().__init__(
            rootnode=rootnode,
            # population_nodes=population_nodes,
            data_cellgroup_col=data_cellgroup_col,
            node_percentages=node_percentages,
            seed=seed,
            include_features=include_features,
            verbose=verbose,
        )

    @staticmethod
    def estimate_population_distribution(node_percentages):
        # From dirichlet package:
        # D : (N, K) shape array
        # ``N`` is the number of observations, ``K`` is the number of
        # parameters for the Dirichlet distribution.
        any_samplewise_zeros = (node_percentages == 0).any(axis=0)
        min_nonzero = node_percentages[node_percentages > 0].min()[0]

        if any(any_samplewise_zeros):
            node_percentages = copy.deepcopy(node_percentages)
            warnings.warn(
                f"Having zero percentages, because of dirichlet estimation replacing zero values with {min_nonzero}"
            )
            # Then there are percentages which are exactly zero.
            # dirichlet estimation cannot cope with those, therefore add a small pseudo-percentage
            for sample_x in any_samplewise_zeros.index:
                if any_samplewise_zeros[sample_x]:
                    # add pseudo-percentage (minimum percentage x 1e-3)
                    node_percentages[sample_x] = (
                        node_percentages[sample_x] + min_nonzero * 1e-3
                    )
                    # re-normalize to a sum of 1
                    node_percentages[sample_x] = node_percentages[sample_x] / sum(
                        node_percentages[sample_x]
                    )
        if "single_sample" in node_percentages.columns:
            node_percentages["single_sample"] = (
                node_percentages["single_sample"] + min_nonzero
            )
            node_percentages["single_sample"] = node_percentages["single_sample"] / sum(
                node_percentages["single_sample"]
            )
            # Add a completely new sample.
            node_percentages["a"] = node_percentages["single_sample"] + (
                min_nonzero * 2
            )
            node_percentages["a"] = node_percentages["a"] / sum(node_percentages["a"])
        # Estimate dirichlet parameters
        alphas = dirichlet.mle(node_percentages.T)
        return {
            "__name": list(node_percentages.index),
            "alpha": pd.DataFrame(alphas, index=node_percentages.index)[0],
        }

    def remove_population(self, population_name: str):
        """Remove a population from the tree

        Args:
            population_name (str): The get_name_full() of a population
        """
        leaf_population_names = self.pop_leafnode_names(population_name)
        for pop_x in leaf_population_names:
            try:
                self.population_parameters["alpha"].drop(pop_x, inplace=True)
                self.population_parameters["__name"].remove(pop_x)
            except KeyError:
                # Then the population was already removed
                pass

    def generate_populations(
        self, population_parameters, n_cells: int, *args, **kwargs
    ):
        onesample_ncells_perpop = self._rng.dirichlet(
            **population_parameters, size=None
        )
        return self.ncells_from_percentages(
            percentages=onesample_ncells_perpop, n_cells=n_cells
        )

    @property
    def precision(self) -> float:
        """Mean from dirichlet distribution

        Estimating a Dirichlet distribution
        Thomas P. Minka
        2000 (revised 2003, 2009, 2012)


        Returns:
            float: Total precision of all populations
        """
        return self.population_parameters["alpha"].sum()

    @property
    def mean_leafs(self) -> pd.Series:
        """Mean from dirichlet distribution

        Estimating a Dirichlet distribution
        Thomas P. Minka
        2000 (revised 2003, 2009, 2012)


        Returns:
            pd.Series: A series (named) of means per cell population
        """
        return self.population_parameters["alpha"] / self.precision

    @property
    def alpha_all(self) -> pd.Series:
        all_alphas = copy.deepcopy(self.population_parameters["alpha"])
        # node_names = list(all_alphas.index)
        for node in anytree.PostOrderIter(self.rootnode_structure):
            if not node.is_leaf:
                current_sum = 0
                for child in node.children:
                    try:
                        current_sum += all_alphas[child.get_name_full()]
                    except KeyError:
                        # Then the population was removed due to lack of cells
                        pass
                all_alphas[node.get_name_full()] = current_sum
        return all_alphas

    def pop_leafnode_names(self, population_node_full_name: Union[str, NBNode]):
        if isinstance(population_node_full_name, str):
            pop_node: NBNode = self.rootnode_structure[population_node_full_name]
        else:
            pop_node = population_node_full_name

        if pop_node.is_leaf:
            return [pop_node.get_name_full()]
        else:
            leafnodes = []
            for x in pop_node.children:
                leafnodes += self.pop_leafnode_names(x)
            return leafnodes

    def pop_alpha(self, population_node_full_name: str):
        this_pop_leafnode_names = self.pop_leafnode_names(
            population_node_full_name=population_node_full_name
        )
        sum_precisions = 0
        for nodename in this_pop_leafnode_names:
            try:
                sum_precisions += self.population_parameters["alpha"][nodename]
            except KeyError:
                pass
        return sum_precisions

    def pop_mean(self, population_node_full_name: str):
        return (
            self.pop_alpha(population_node_full_name=population_node_full_name)
            / self.precision
        )

    def new_pop_mean(self, population_node_full_name: str, percentage: float):
        if percentage < 0 or percentage > 1:
            raise ValueError("percentage must be between 0 and 1")

        new_pop_alpha = percentage * self.precision
        new_other_alpha = (1 - percentage) * self.precision

        pop_leaf_names = self.pop_leafnode_names(
            population_node_full_name=population_node_full_name
        )
        other_leaf_names = [x for x in self.mean_leafs.index if x not in pop_leaf_names]

        for names, newalpha in zip(
            [pop_leaf_names, other_leaf_names],
            [new_pop_alpha, new_other_alpha],
        ):
            leaf_means = {}
            for x in names:
                try:
                    leaf_means[x] = self.mean_leafs[x]
                except KeyError:
                    # Then mean_leafs[x] was not accessible
                    # -->  no points have been in this population during initialization
                    pass
            leaf_means_sum = sum(leaf_means.values())
            leaf_means_portion = {
                key: value / leaf_means_sum for key, value in leaf_means.items()
            }
            for leaf_popname, leaf_portion in leaf_means_portion.items():
                self.population_parameters["alpha"][leaf_popname] = (
                    newalpha * leaf_portion
                )
