""" This module contains probability functions for use when
    inferring the (distance modulus, log-extinction, R_V).
"""

import numpy as np


def photom_prob(params, star):
    """ emcee_prob(params, star)

        Finds the likelihood of a set of stellar params,
        for a given star_posterior object (which contains
        the observations and on-sky position of the star).

        This function only uses the observed photometry of a
        star.

        If ([M/H], T_eff, log(g)) is outside the region covered
        by star's iso_obj the a log probability of -infinity is
        returned.

        Parameters
        ----------
        params : ndarray(float)
            The stellar parameters for which we desire the
            likelihood. Its contents are (in order):
            [M/H], T_eff, log(g), distance, A_4000
            R_5495.
        star : star_posterior
            The object that contains the observations, on-sky
            position (as (l,b)) and other information about
            the star in question.

        Returns
        -------
        prob : float
            The log likelihood of the parameter set params
            for the star described by star.
    """

#    try:
    iso_obj = star.isochrones.query(params[0], params[1], params[2])
#    except IndexError:
#        return -np.inf+np.zeros(params.shape[1])

    R_out_of_bounds = np.logical_or(params[5] < 2.05, params[5] > 5.05)
    params[5, R_out_of_bounds] = 3.1

    try:
        a_out_of_bounds = params[4] < -5
        params[4, a_out_of_bounds] = -5

        A = np.exp(params[4])
        dist = np.power(10., params[3]/5.+1.)
        R_gal = np.sqrt(8000*8000 + np.power(dist*star.cosb, 2)
                        - 2*8000*dist*star.cosb*star.cosl)
        prob = (np.log(iso_obj.Jac) - 2.3*np.log(iso_obj.Mi)
                + 2.3026*iso_obj.logage
                - np.power(params[0]+(R_gal-8000.)*0.00006, 2)/(2*0.04))

        for band in star.mag.keys():
            prob -= (np.power(star.mag[band] - (iso_obj.abs_mag[band]
                              + params[3] + iso_obj.AX(band, A, params[5])), 2)
                     / (2*star.d_mag[band]*star.d_mag[band]))

        prob[iso_obj.Jac == 0] = -np.inf
        prob[np.logical_or(params[5] < 2.05, params[5] > 5.05)] = -np.inf
        prob[params[4] < -5] = -np.inf

        all_out_of_bounds = np.logical_or(R_out_of_bounds, a_out_of_bounds)
        prob[all_out_of_bounds] = -np.inf

        return prob

    except OverflowError:
        return -np.inf
