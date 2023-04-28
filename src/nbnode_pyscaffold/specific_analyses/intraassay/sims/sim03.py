from typing import Union


from nbnode_pyscaffold.simulation.FlowSimulationTree import FlowSimulationTreeDirichlet
from nbnode_pyscaffold.simulation.TreeMeanDistributionSampler import TreeMeanDistributionSampler
from nbnode_pyscaffold.simulation.TreeMeanDistributionSampler import PseudoTorchDistributionNormal

def sim03_m_sd(
    flowsim_tree: Union[str, FlowSimulationTreeDirichlet],
    meanshift: float,
    sd: float,
    population_name: str = "/AllCells/CD4+/CD8-/Tem",
    n_samples=100,
    n_cells=10000,
    use_only_diagonal_covmat=False,
    verbose=True,
    seed_sample_0=129873,
    save_dir="sim/sim03_m_sd",
    only_return_sampled_cell_numbers=False,
):
    generator = TreeMeanDistributionSampler(
        population_name_to_change=population_name,
        mean_distribution=lambda original_mean: PseudoTorchDistributionNormal(
            loc=original_mean + meanshift, scale=sd
        ),
        # import torch.distributions as D
        # mean_distribution=lambda original_mean: D.Normal(
        #     loc=original_mean + meanshift, scale=sd
        # ),
        flowsim_tree=flowsim_tree,
        n_samples=n_samples,
        n_cells=n_cells,
        use_only_diagonal_covmat=use_only_diagonal_covmat,
        verbose=verbose,
        seed_sample_0=seed_sample_0,
        save_dir=save_dir,
        only_return_sampled_cell_numbers=only_return_sampled_cell_numbers,
    )
    return generator.sample()