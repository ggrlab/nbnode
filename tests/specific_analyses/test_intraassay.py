from unittest import TestCase
import copy
import os
import pickle

import numpy as np
from nbnode.testutil.helpers import find_dirname_above_currentfile
from nbnode.specific_analyses.intraassay.gate_init import gate_init
from nbnode.io.pickle_open_dump import pickle_open_dump

TESTS_DIR = find_dirname_above_currentfile()


class TestIntraassayData(TestCase):
    @classmethod
    def setUpClass(self):
        try:
            with open(
                "examples/results/intraassay_gate_init_4samples.pickle", "rb"
            ) as f:
                self.celltree, node_counts_df, self.flowsim_tree = pickle.load(f)
        except:
            # 4 samples just to speed up things
            self.celltree, node_counts_df, self.flowsim_tree = gate_init(
                sample_list=[0, 1, 2, 3]
            )
            pickle_open_dump(
                (self.celltree, node_counts_df, self.flowsim_tree),
                "examples/results/intraassay_gate_init_4samples.pickle",
            )

    def test_sample_cells(self):
        self.flowsim_tree.set_seed(10289)
        z = self.flowsim_tree.sample(n_cells=100)
        assert z.shape == (100, 13)

    def test_sample_populations(self):
        self.flowsim_tree.set_seed(10289)
        # sample_populations() gets one complete sample of n_cells split across all populations
        n_cells = 1e10
        overconfident_flowdist = copy.deepcopy(self.flowsim_tree)
        overconfident_flowdist.population_parameters["alpha"] *= 1e8
        many_cells_overconfident = overconfident_flowdist.sample_populations(
            n_cells=n_cells
        )
        assert np.allclose(
            (many_cells_overconfident / (1.0 * n_cells)),
            self.flowsim_tree.mean_leafs,
            atol=1e-5,
            rtol=1e-5,
        )

    def test_changepop(self):
        self.flowsim_tree.set_seed(10289)
        n_cells = 1e6
        overconfident_flowdist = copy.deepcopy(self.flowsim_tree)
        overconfident_flowdist.population_parameters["alpha"] *= 1e8
        many_cells_overconfident = overconfident_flowdist.sample_populations(
            n_cells=n_cells
        )
        assert np.allclose(
            (many_cells_overconfident / (1.0 * n_cells)),
            self.flowsim_tree.mean_leafs,
            atol=1e-5,
            rtol=1e-5,
        )

    def test_sim00_baseline(self):
        import nbnode.specific_analyses.intraassay.sims as ia_sims

        (
            true_popcounts_sim00,
            changed_parameters,
            generated_samples,
        ) = ia_sims.sim00_baseline(
            flowsim_tree=self.flowsim_tree,
            n_samples=5,
            n_cells=100,
            use_only_diagonal_covmat=True,
            save_dir=None,
            seed_sample_0=1392857,
        )

        (
            true_popcounts_sim00_v2,
            changed_parameters_v2,
            generated_samples_v2,
        ) = ia_sims.sim00_baseline(
            flowsim_tree=self.flowsim_tree,
            n_samples=5,
            n_cells=100,
            use_only_diagonal_covmat=True,
            save_dir=None,
            seed_sample_0=1392857,
        )

        # Check that generating with the same seed leads to the exact same results
        assert true_popcounts_sim00_v2.equals(true_popcounts_sim00)
        assert changed_parameters_v2["__name"] == changed_parameters["__name"]
        assert changed_parameters_v2["alpha"].equals(changed_parameters["alpha"])
        assert all(
            [x.equals(y) for x, y in zip(generated_samples_v2, generated_samples)]
        )

        (
            true_popcounts_sim00_v3,
            changed_parameters_v3,
            generated_samples_v3,
        ) = ia_sims.sim00_baseline(
            flowsim_tree=self.flowsim_tree,
            n_samples=5,
            n_cells=100,
            use_only_diagonal_covmat=True,
            save_dir=None,
            seed_sample_0=1392852,
        )

        # Compared to the first initialization,
        #   The number of cells must be different per population
        assert not true_popcounts_sim00_v3.equals(true_popcounts_sim00)
        #   The parameters names must be the same
        assert changed_parameters_v3["__name"] == changed_parameters["__name"]
        #   The parameters values must be the same
        assert changed_parameters_v3["alpha"].equals(changed_parameters["alpha"])
        #   The actually generated cells must be different
        assert not all(
            [x.equals(y) for x, y in zip(generated_samples_v3, generated_samples)]
        )

        (
            true_popcounts_sim00_v4,
            changed_parameters_v4,
            generated_samples_v4,
        ) = ia_sims.sim00_baseline(
            flowsim_tree=self.flowsim_tree,
            n_samples=5,
            n_cells=100,
            use_only_diagonal_covmat=False,
            save_dir=None,
            seed_sample_0=1392852,
        )
        # Compared to v3, the number of cells must be identical
        #   The number of cells must be identical per population
        assert true_popcounts_sim00_v3.equals(true_popcounts_sim00_v4)
        #   The parameters names must be the same
        assert changed_parameters_v3["__name"] == changed_parameters_v4["__name"]
        #   The parameters values must be the same
        assert changed_parameters_v3["alpha"].equals(changed_parameters_v4["alpha"])
        #   The actually generated cells must be different
        assert not all(
            [x.equals(y) for x, y in zip(generated_samples_v3, generated_samples_v4)]
        )

    def test_simulations(self):
        import nbnode.specific_analyses.intraassay.sims as ia_sims

        n_cells = 1e5
        n_samples = 5

        done_sim00 = ia_sims.sim00_baseline(
            flowsim_tree=self.flowsim_tree,
            n_cells=n_cells,
            n_samples=n_samples,
            save_dir=None,
        )
        # there must be ``n_samples`` samples
        assert len(done_sim00[2]) == n_samples
        assert done_sim00[0].shape[1] == n_samples
        # For every sample there must be ``n_cells`` cells and 13 parameters
        assert all([x.shape == tuple((n_cells, 13)) for x in done_sim00[2]])

        done_sim01 = ia_sims.sim01_double_tcm(
            flowsim_tree=self.flowsim_tree,
            n_cells=n_cells,
            n_samples=n_samples,
            save_dir=None,
        )
        # there must be ``n_samples`` samples
        assert len(done_sim01[2]) == n_samples
        assert done_sim01[0].shape[1] == n_samples
        # For every sample there must be ``n_cells`` cells and 13 parameters
        assert all([x.shape == tuple((n_cells, 13)) for x in done_sim01[2]])

        done_sim02 = ia_sims.sim02_temra(
            flowsim_tree=self.flowsim_tree,
            n_cells=n_cells,
            n_samples=n_samples,
            save_dir=None,
        )
        # there must be ``n_samples`` samples
        assert len(done_sim02[2]) == n_samples
        assert done_sim02[0].shape[1] == n_samples
        # For every sample there must be ``n_cells`` cells and 13 parameters
        assert all([x.shape == tuple((n_cells, 13)) for x in done_sim02[2]])

    def test_visualinspection_simulations_cellnumbers(self):
        import pickle

        try:
            with open("examples/results/intraassay_gate_init.pickle", "rb") as f:
                full_celltree, node_counts_df, full_flowsim_tree = pickle.load(f)
        except:
            full_flowsim_tree = self.flowsim_tree

        import matplotlib.pyplot as plt

        import nbnode.specific_analyses.intraassay.sims as ia_sims
        from nbnode.utils.merge_leaf_nodes import merge_leaf_nodes

        relevant_pops = [
            "/AllCells/CD4+/CD8-/Tcm",
            "/AllCells/CD4+/CD8-/Tem",
            "/AllCells/CD4-/CD8+/Temra",
        ]
        n_cells = 1e4
        n_samples = 1000

        done_sim00 = ia_sims.sim00_baseline(
            flowsim_tree=full_flowsim_tree,
            n_cells=n_cells,
            n_samples=n_samples,
            save_dir=None,
            debugging_only_return_sampled_cell_numbers=True,
        )
        # there must be ``n_samples`` samples
        assert done_sim00[0].shape[1] == n_samples
        relevant_pop_sim00 = {
            relevant_intermediate_population: merge_leaf_nodes(
                relevant_intermediate_population, done_sim00[0]
            )
            / n_cells
            for relevant_intermediate_population in relevant_pops
        }

        done_sim01 = ia_sims.sim01_double_tcm(
            flowsim_tree=full_flowsim_tree,
            n_cells=n_cells,
            n_samples=n_samples,
            save_dir=None,
            debugging_only_return_sampled_cell_numbers=True,
        )
        # there must be ``n_samples`` samples
        assert done_sim01[0].shape[1] == n_samples
        relevant_pop_sim01 = {
            relevant_intermediate_population: merge_leaf_nodes(
                relevant_intermediate_population, done_sim01[0]
            )
            / n_cells
            for relevant_intermediate_population in relevant_pops
        }
        assert (
            relevant_pop_sim00["/AllCells/CD4+/CD8-/Tcm"].mean() * 2
            - relevant_pop_sim01["/AllCells/CD4+/CD8-/Tcm"].mean()
            < 1e-3
        )

        done_sim02 = ia_sims.sim02_temra(
            flowsim_tree=full_flowsim_tree,
            n_cells=n_cells,
            n_samples=n_samples,
            save_dir=None,
            debugging_only_return_sampled_cell_numbers=True,
        )
        # there must be ``n_samples`` samples
        assert done_sim02[0].shape[1] == n_samples
        relevant_pop_sim02 = {
            relevant_intermediate_population: merge_leaf_nodes(
                relevant_intermediate_population, done_sim02[0]
            )
            / n_cells
            for relevant_intermediate_population in relevant_pops
        }

        # Inspect the following figures.
        # They should all look nicely

        ax = relevant_pop_sim00["/AllCells/CD4+/CD8-/Tcm"].hist()
        fig = ax.get_figure()
        ax = relevant_pop_sim01["/AllCells/CD4+/CD8-/Tcm"].hist()
        fig = ax.get_figure()
        fig.savefig("removeme_relevant_pop_Tcm_double_sim01_vs_02.pdf")
        plt.close("all")

        ax = relevant_pop_sim00["/AllCells/CD4-/CD8+/Temra"].hist()
        fig = ax.get_figure()
        ax = relevant_pop_sim01["/AllCells/CD4-/CD8+/Temra"].hist()
        fig = ax.get_figure()
        ax = relevant_pop_sim02["/AllCells/CD4-/CD8+/Temra"].hist()
        fig = ax.get_figure()
        fig.savefig("removeme_relevant_pop_Temra_sim00-03.pdf")
        plt.close("all")

        ax = relevant_pop_sim00["/AllCells/CD4+/CD8-/Tem"].hist()
        fig = ax.get_figure()
        ax = relevant_pop_sim01["/AllCells/CD4+/CD8-/Tem"].hist()
        fig = ax.get_figure()
        fig.savefig("removeme_relevant_pop_Tem_double_sim01_vs_02.pdf")
        plt.close("all")

    def test_visualinspection_sim03_vary_sd(self):
        try:
            with open("examples/results/intraassay_gate_init.pickle", "rb") as f:
                full_celltree, node_counts_df, full_flowsim_tree = pickle.load(f)
        except:
            full_flowsim_tree = self.flowsim_tree

        import matplotlib.pyplot as plt
        import nbnode.specific_analyses.intraassay.sims as ia_sims

        from nbnode.utils.merge_leaf_nodes import merge_leaf_nodes

        relevant_pops = [
            "/AllCells/CD4+/CD8-/Tem",
        ]

        n_cells = 1e4
        n_samples = 50
        done_sims = {}
        for sd in [1e-7, 1, 10]:
            done_sims[f"sim03_sd{sd}"] = ia_sims.sim03_m_sd(
                flowsim_tree=full_flowsim_tree,
                population_name="/AllCells/CD4+/CD8-/Tem",
                meanshift=0,
                sd=sd,
                n_cells=n_cells,
                n_samples=n_samples,
                save_dir=None,
                debugging_only_return_sampled_cell_numbers=True,
            )
            assert done_sims[f"sim03_sd{sd}"][0].shape[1] == n_samples
        import pandas as pd

        relevant_pop_sim03 = {
            key: pd.DataFrame.from_dict(
                {
                    relevant_intermediate_population: merge_leaf_nodes(
                        relevant_intermediate_population, done_simulation[0]
                    )
                    / n_cells
                    for relevant_intermediate_population in relevant_pops
                }
            )
            for key, done_simulation in done_sims.items()
        }
        relevant_pop_sim03["sim03_sd1e-07"]
        relevant_pop_sim03["sim03_sd1"]
        relevant_pop_sim03["sim03_sd10"]
        # Inspect the following figures.
        # They should all look nicely

        ax = relevant_pop_sim03["sim03_sd1e-07"]["/AllCells/CD4+/CD8-/Tem"].hist(
            edgecolor="None", alpha=0.5, label="sim03_sd1e-07"
        )
        fig = ax.get_figure()
        ax = relevant_pop_sim03["sim03_sd1"]["/AllCells/CD4+/CD8-/Tem"].hist(
            edgecolor="None", alpha=0.5, label="sim03_sd1"
        )
        fig = ax.get_figure()
        ax = relevant_pop_sim03["sim03_sd10"]["/AllCells/CD4+/CD8-/Tem"].hist(
            edgecolor="None", alpha=0.5, legend="sim03_sd10"
        )
        fig = ax.get_figure()
        fig.savefig("removeme_relevant_pop_Tem_vary_SD.pdf")
        plt.close("all")

    def test_visualinspection_sim03_vary_mean(self):
        try:
            with open("examples/results/intraassay_gate_init.pickle", "rb") as f:
                full_celltree, node_counts_df, full_flowsim_tree = pickle.load(f)
        except:
            full_flowsim_tree = self.flowsim_tree

        import matplotlib.pyplot as plt
        import nbnode.specific_analyses.intraassay.sims as ia_sims

        from nbnode.utils.merge_leaf_nodes import merge_leaf_nodes

        relevant_pops = [
            "/AllCells/CD4+/CD8-/Tem",
        ]
        meanshifts = [0, 1, 2, 5, 10]
        n_cells = 1e4
        n_samples = 50
        done_sims = {}
        for meanshift in meanshifts:
            done_sims[f"sim03_m.{meanshift}_sd.1"] = ia_sims.sim03_m_sd(
                flowsim_tree=full_flowsim_tree,
                population_name="/AllCells/CD4+/CD8-/Tem",
                meanshift=meanshift,
                sd=1,
                n_cells=n_cells,
                n_samples=n_samples,
                save_dir=None,
                debugging_only_return_sampled_cell_numbers=True,
            )
            assert done_sims[f"sim03_m.{meanshift}_sd.1"][0].shape[1] == n_samples
        import pandas as pd

        relevant_pop_sim03 = {
            key: pd.DataFrame.from_dict(
                {
                    relevant_intermediate_population: merge_leaf_nodes(
                        relevant_intermediate_population, done_simulation[0]
                    )
                    / n_cells
                    for relevant_intermediate_population in relevant_pops
                }
            )
            for key, done_simulation in done_sims.items()
        }

        for meanshift in meanshifts[1:]:
            fig = plt.figure(figsize=[15, 7])
            ax = relevant_pop_sim03[f"sim03_m.{meanshift}_sd.1"][
                "/AllCells/CD4+/CD8-/Tem"
            ].hist(edgecolor="None", alpha=0.5, label=f"sim03_m.{meanshift}_sd.1")
            ax.set_xlim([0, 0.4])

            ax = relevant_pop_sim03[f"sim03_m.{meanshifts[0]}_sd.1"][
                "/AllCells/CD4+/CD8-/Tem"
            ].hist(edgecolor="None", alpha=0.5, legend=True)
            ax.set_title(f"Last element: sim03_m.{meanshifts[0]}_sd.1")
            ax.set_xlim([0, 0.4])
            fig.savefig(f"removeme_relevant_pop_Tem_vary_mean_{meanshift}.pdf")
            plt.close("all")

    def test_visualinspection_sim03_vary_mean_sd(self):
        try:
            with open("examples/results/intraassay_gate_init.pickle", "rb") as f:
                full_celltree, node_counts_df, full_flowsim_tree = pickle.load(f)
        except:
            full_flowsim_tree = self.flowsim_tree

        import matplotlib.pyplot as plt
        import nbnode.specific_analyses.intraassay.sims as ia_sims

        from nbnode.utils.merge_leaf_nodes import merge_leaf_nodes

        relevant_pops = [
            "/AllCells/CD4+/CD8-/Tem",
        ]
        meanshifts = [0, 1, 5, 10]
        n_cells = 1e4
        n_samples = 50
        done_sims = {}
        for meanshift in meanshifts:
            done_sims[f"sim03_m.{meanshift}_sd.1"] = ia_sims.sim03_m_sd(
                flowsim_tree=full_flowsim_tree,
                population_name="/AllCells/CD4+/CD8-/Tem",
                meanshift=meanshift,
                sd=1,
                n_cells=n_cells,
                n_samples=n_samples,
                save_dir=None,
                debugging_only_return_sampled_cell_numbers=True,
            )
            assert done_sims[f"sim03_m.{meanshift}_sd.1"][0].shape[1] == n_samples
            done_sims[f"sim03_m.{meanshift}_sd.5"] = ia_sims.sim03_m_sd(
                flowsim_tree=full_flowsim_tree,
                population_name="/AllCells/CD4+/CD8-/Tem",
                meanshift=meanshift,
                sd=5,
                n_cells=n_cells,
                n_samples=n_samples,
                save_dir=None,
                debugging_only_return_sampled_cell_numbers=True,
            )
        import pandas as pd

        relevant_pop_sim03 = {
            key: pd.DataFrame.from_dict(
                {
                    relevant_intermediate_population: merge_leaf_nodes(
                        relevant_intermediate_population, done_simulation[0]
                    )
                    / n_cells
                    for relevant_intermediate_population in relevant_pops
                }
            )
            for key, done_simulation in done_sims.items()
        }

        for sim_name in list(done_sims.keys())[1:]:
            fig = plt.figure(figsize=[15, 7])
            ax = relevant_pop_sim03[sim_name]["/AllCells/CD4+/CD8-/Tem"].hist(
                edgecolor="None", alpha=0.5, label=sim_name
            )
            ax.set_xlim([0, 0.4])

            ax = relevant_pop_sim03[list(done_sims.keys())[0]][
                "/AllCells/CD4+/CD8-/Tem"
            ].hist(edgecolor="None", alpha=0.5, legend=True)
            ax.set_title(f"Last element: {list(done_sims.keys())[0]}")
            ax.set_xlim([0, 0.4])
            fig.savefig(f"removeme_relevant_pop_Tem_{sim_name}.pdf")
            plt.close("all")
