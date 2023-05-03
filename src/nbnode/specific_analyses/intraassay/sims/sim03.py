from typing import Union

from nbnode.simulation.FlowSimulationTree import FlowSimulationTreeDirichlet
from nbnode.simulation.TreeMeanDistributionSampler import (
    PseudoTorchDistributionNormal,
    TreeMeanDistributionSampler,
)


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
    """Create a target normal distribution of a population

    Generate a normal distribution with mean=original_mean+meanshift and standard
    deviation=sd. Then, sample from this distribution to create a new value
    for the population. The population's concentration parameter is then set to
    the concentration parameter corresponding to the new value.

    Args:
        flowsim_tree (Union[str, FlowSimulationTreeDirichlet]):
            See ``TreeMeanDistributionSampler``.
        meanshift (float):
            Mean shift of the target normal distribution. The mean of the target normal
            distribution is original_mean + meanshift. E.g.::

                PseudoTorchDistributionNormal(
                    loc=original_mean + meanshift, scale=sd
                )


        sd (float):
            Standard deviation of the target normal distribution. E.g.::

                PseudoTorchDistributionNormal(
                    loc=original_mean + meanshift, scale=sd
                )


        population_name (str, optional):
            The get_name_full() of the population to change.
            Defaults to "/AllCells/CD4+/CD8-/Tem".
        n_samples (int, optional):
            The number of samples to generate.
            Defaults to 100.
        n_cells (int, optional):
            The number of cells per sample.
            Defaults to 10000.
        use_only_diagonal_covmat (bool, optional):
            Whether to use only the diagonal of the covariance matrix.
            Defaults to False.
        verbose (bool, optional):
            Whether to print progress.
            Defaults to True.
        seed_sample_0 (int, optional):
            The seed for the first sample. All further sample seeds are incremented by
            1 per sample.
            Defaults to 129873.
        save_dir (str, optional):
            The directory to save the samples to.
            Defaults to "sim/sim03_m_sd".

        only_return_sampled_cell_numbers (bool, optional):
            Whether to only return the sampled cell numbers.
            Defaults to False.

    Returns:
        generator.sample(): generator.sample()
    """
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
