from unittest import TestCase
import os

import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import matplotlib.pyplot as plt
# from ccc.plot.plotting import plot_samples
from nbnode_pyscaffold.simulation.MixtureDistribution import (
    vary_single_percentage,
    PercentageMixtureDistribution,
)


class Test(TestCase):
    def test_PercentageMixtureDistribution(self):
        means_A = torch.tensor([5, 1]).float()
        covmat_A = torch.eye(len(means_A)).float()
        a = PercentageMixtureDistribution(
            mixture_distribution=[
                MultivariateNormal(
                    loc=means_A + 10,  # mean of the distribution ("location"?)
                    covariance_matrix=covmat_A,  #
                ),
                MultivariateNormal(loc=means_A, covariance_matrix=covmat_A),  #
                MultivariateNormal(
                    loc=torch.tensor([-2, 1]).float(), covariance_matrix=covmat_A  #
                ),
            ],
            component_percentage=[0.01, 0.7, 0.2],
        )  # these are not percentages but odds! (They do not have to sum up to 1)

        # Here should now be 3 clusters:
        # bottom, bit right, most points: second distribution with means [5, 1]
        # bottom, left, middle amount of points: third distribution with means [-2, 1]
        # top, right, least amount of points: third distribution with means [15, 11]
        # plot_samples(cloud_list=[a.sample(10000)], marker_x=0, marker_y=1)
        # plt.show()  # outcommented because the CI cannot test an image.

    def test_vary_single_percentage(self):
        true_new_odds = torch.tensor(
            [[0.0100, 0.9900], [0.1000, 0.9000], [0.2, 0.8], [0.5, 0.5]]
        )
        base_cluster_odds = torch.tensor([1, 1]).float()
        for index, missing_cluster_percentage in enumerate([0.01, 0.1, 0.2, 0.5]):
            assert torch.allclose(
                vary_single_percentage(
                    odds=base_cluster_odds,
                    target_percentage=missing_cluster_percentage,
                    index=0,
                ),
                true_new_odds[index],
            )
            # If I take the index 1, the second index gets the new percentages,
            # therefore the true new odds are exactly the same just reversed
            #   --> torch.flip([0])
            assert torch.allclose(
                vary_single_percentage(
                    odds=base_cluster_odds,
                    target_percentage=missing_cluster_percentage,
                    index=1,
                ),
                true_new_odds[index].flip([0]),
            )

    def test_PercentageMixtureDistribution_percentage_zero(self):
        means_A = torch.tensor([5, 1]).float()
        covmat_A = torch.eye(len(means_A)).float()
        a = PercentageMixtureDistribution(
            mixture_distribution=[
                MultivariateNormal(
                    loc=means_A + 10,  # mean of the distribution ("location"?)
                    covariance_matrix=covmat_A,  #
                ),
                MultivariateNormal(loc=means_A, covariance_matrix=covmat_A),  #
                MultivariateNormal(
                    loc=torch.tensor([-2, 1]).float(), covariance_matrix=covmat_A  #
                ),
            ],
            component_percentage=[0, 0.7, 0.2],
        )  # these are not percentages but odds! (They do not have to sum up to 1)
        assert len(a.sample(10000, return_counts=True)[1]) == 3
