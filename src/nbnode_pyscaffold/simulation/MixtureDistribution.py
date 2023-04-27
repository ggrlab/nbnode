import torch
from typing import List, Union


def vary_single_percentage(
    odds: Union[torch.Tensor, List], index: int = 0, target_percentage: float = 0.2
):
    """
    Given a odds vector, after normalization to 1, the `index` must have `target_percentage`
    Args:
        odds:
            The previous "percentages", do not have to sum up to 1, therefore I call it odds.
        index:
            After normalization, this index of `odds` becomes the target percentage
        target_percentage:
            After normalization, the odds value is exactly this target_percentage.

    Returns:

    """
    odds[0] = float(odds[0])  # so I make sure that odds is completely float
    if not isinstance(odds, torch.Tensor):
        odds = torch.tensor(odds)

    odds_without_index = torch.cat([odds[:index], odds[index + 1 :]])
    backcalc_ratio = (
        target_percentage * odds_without_index.abs().sum() / (1 - target_percentage)
    )
    component_percentage = odds
    component_percentage[index] = backcalc_ratio
    return component_percentage / component_percentage.abs().sum()


class PercentageMixtureDistribution(object):
    def __init__(
        self,
        mixture_distribution: Union[List, torch.Tensor],
        component_percentage: Union[List, torch.Tensor],
    ):
        if len(mixture_distribution) != len(component_percentage):
            raise ValueError("Must be a percentage for each mixture")
        self._mix = mixture_distribution
        if not isinstance(component_percentage, torch.Tensor):
            component_percentage = torch.tensor(component_percentage)
        self._perc = component_percentage
        self._perc = self._perc / self._perc.abs().sum()  # normalize to 1

    def __len__(self):
        return len(self._mix)

    def sample(self, n, return_counts: bool = False, return_only_counts: bool = False):
        # return_counts = True yields an additional tensor where
        # the first element is the count for the first unique value
        # the second element is the count for the second unique value
        # ...
        sampled = self.unif_sampling(self._perc, n)
        how_many_per_component = [
            (component_i == sampled).sum() for component_i in range(len(self))
        ]
        if return_only_counts:
            return how_many_per_component

        if not return_counts:
            return torch.cat(
                [
                    self._mix[index].sample([count_per_component])
                    for index, count_per_component in enumerate(how_many_per_component)
                ]
            )
        else:
            return (
                torch.cat(
                    [
                        self._mix[index].sample([count_per_component])
                        for index, count_per_component in enumerate(
                            how_many_per_component
                        )
                    ]
                ),
                how_many_per_component,
            )

    @staticmethod
    def unif_sampling(percentages: torch.Tensor, n: int):
        percentages /= percentages.abs().sum()  # normalize to 1
        perc_cumsum = torch.cumsum(percentages, 0)
        unif_sample = torch.rand(n)  # uniform sampling [0, 1)

        not_already_found = torch.BoolTensor(torch.ones((len(unif_sample)), dtype=bool))
        which_components = torch.zeros((len(unif_sample)), dtype=int)
        for index_perc, perc_summed in enumerate(perc_cumsum):
            unif_smaller_and_not_yet_found = (
                unif_sample < perc_summed
            ) & not_already_found
            which_components[unif_smaller_and_not_yet_found] = index_perc
            not_already_found[unif_smaller_and_not_yet_found] = False
        return which_components
