"""
Author : Chrissy Aceves
Email : cmaceves@scripps.edu
"""

import os 
import sys
import math


def maximum_likelihood(mu_list, sigma_list, pi_list, frequencies):
    """
    Given a mu, sigma, pi and frequency value calcualte the maximum liklihood of a point
    belonging to a group.

    Parameters
    -----------
    mu_list : list
    sigma_list : list 
    pi_list : list
    frequencies : list

    Returns
    -------
    cluster : list
    likelihood : list
    all_likelihoods : list
    """

    likelihood = []
    cluster = []
    all_likelihoods = []

    for point in frequencies:
        tmp_likelihood = []
        for mu, sigma, pi in zip(mu_list, sigma_list, pi_list):
            term1 = (point - mu) ** 2
            term1 = term1 * -0.5
            term1 = term1 / sigma
            term1 = math.exp(term1)            
            term1 = term1 * pi

            term2 = 2 * math.pi
            term2 = term2 * (sigma)
            term2 = 1 / (term2 ** (0.5))
            l = term1 * term2
            tmp_likelihood.append(l)

        all_likelihoods.append(tmp_likelihood)
        if max(tmp_likelihood) > 0:
            likelihood.append(max(tmp_likelihood))
            loc = tmp_likelihood.index(max(tmp_likelihood))
            cluster.append(mu_list[loc])
        else:
            cluster.append(-1)
            likelihood.append(-1)
    return(cluster, likelihood, all_likelihoods)
