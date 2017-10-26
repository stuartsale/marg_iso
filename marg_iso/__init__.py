# Import stuff

from __future__ import print_function, division
import emcee
import math
import numpy as np
from scipy import linalg
import sklearn as sk
import sklearn.cluster as sk_c
import sklearn.mixture as sk_m

# Attempt to import matplotlib
try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl_present = True
except ImportError:
    mpl_present = False

import isolib as il
from gmm_extra import MeanCov_GMM


def emcee_prob(params, star):
    """ emcee_prob(params, star)

        Finds the likelihood of a set of stellar params,
        for a given star_posterior object (which contains
        the observations and on-sky position of the star).

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


class star_posterior:

    """ This is a class to claculate and contain the marginal
        posterior distribution on a star's distance modulus,
        ln extinction and extinction law, i.e. p(s,lnA,R|D).

        The class employs a special, vectorised version of emcee
        to sample from the posterior. Marginalisation over Teff,
        logg and metallicity is performed by simply ignoring these
        values in the chain.
    """

    # init function

    def __init__(self, gal_l, gal_b, mag_in, d_mag_in, isochrones=None,
                 isochrone_file=None, init_bands=None):
        """ __init__(l, b, mag_in, d_mag_in, isochrones=None,
                     isochrone_file=None, init_bands=None)

            Initialise a star_posterior object, giving it the
            on-sky position of the star and some photometry
            (with uncertainties).

            Parameters
            ----------
            gal_l : float
                The Galactic Longitude of the star.
            gal_b : float
                The Galactic lattitude of the star
            mag_in : dict
                The photometry of the star, where the band label is
                the key and the magnitude is the value.
            d_mag_in : dict
                The photometric uncertainties, where the band label
                is the key and the uncertainty is the value.
            isochrones : isolib.iso_grid_tefflogg, optional
                A set of isochrones in ([Fe/H], Teff, logg) space.
            isochrones_file : string, optional
                The location of a file that contains isochrones
                that can be read in by isolib for use.
            init_bands : list, optional
                A (sub-)set of the observed photometric bands
                which are used to make an initial guess at the
                parameters of the star. Cannot contain bands that
                are not in mag_in.

            Notes
            -----
            For a list of vaild photometric bands see <INSERT
            ADDRESS>.

            Either isochrones or isochrone_file must be provided.
            Otherwise the class will have no ischrone information.
        """

        self.colors = np.array([x for x in 'bgrcmybgrcmybgrcmybgrcmy'])
        self.colors = np.hstack([self.colors] * 20)

        self.gal_l = np.radians(gal_l)
        self.gal_b = np.radians(gal_b)

        self.sinl = np.sin(self.gal_l)
        self.cosl = np.cos(self.gal_l)
        self.sinb = np.sin(self.gal_b)
        self.cosb = np.cos(self.gal_b)

        self.mag = mag_in
        self.d_mag = d_mag_in

        if init_bands is None:
            self.init_bands = self.mag.keys()[:2]
        else:
            self.init_bands = init_bands

        if isochrones is None:
            if isochrone_file is not None:
                isochrones = il.iso_grid_tefflogg(isochrone_file,
                                                  bands=self.mag.keys())
            else:
                raise IOError("Either an isochrone must be provided "
                              "or a filename for one given")
        self.isochrones = isochrones

        self.MCMC_run = False
        self.best_gmm = None

    # ==============================================================
    # Functions to work with emcee sampler

    def emcee_init(self, N_walkers, chain_length):
        """ emcee_init(N_walkers, chain_length)

            Initialises the emcee walkers.

            Parameters
            ----------
            N_walkers : int
                The number of walkers to be used.
            chain_length: int
                The length of the MCMC chain
        """

        self.start_params = np.zeros([N_walkers, 6])

        guess_set = []
        guess_set.append([0., 3.663, 4.57, 0., 0., 3.1])    # K4V
        guess_set.append([0., 3.672, 4.56, 0., 0., 3.1])    # K3V
        guess_set.append([0., 3.686, 4.55, 0., 0., 3.1])    # K2V
        guess_set.append([0., 3.695, 4.55, 0., 0., 3.1])    # K1V
        guess_set.append([0., 3.703, 4.57, 0., 0., 3.1])    # K0V
        guess_set.append([0., 3.720, 4.55, 0., 0., 3.1])    # G8V
        guess_set.append([0., 3.740, 4.49, 0., 0., 3.1])    # G5V
        guess_set.append([0., 3.763, 4.40, 0., 0., 3.1])    # G2V
        guess_set.append([0., 3.774, 4.39, 0., 0., 3.1])    # G0V
        guess_set.append([0., 3.789, 4.35, 0., 0., 3.1])    # F8V
        guess_set.append([0., 3.813, 4.34, 0., 0., 3.1])    # F5V
        guess_set.append([0., 3.845, 4.26, 0., 0., 3.1])    # F2V
        guess_set.append([0., 3.863, 4.28, 0., 0., 3.1])    # F0V
        guess_set.append([0., 3.903, 4.26, 0., 0., 3.1])    # A7V
        guess_set.append([0., 3.924, 4.22, 0., 0., 3.1])    # A5V
        guess_set.append([0., 3.949, 4.20, 0., 0., 3.1])    # A3V
        guess_set.append([0., 3.961, 4.16, 0., 0., 3.1])    # A2V

#        guess_set.append([0.,3.763 ,3.20 ,0.,0.,3.1])    #G2III
#        guess_set.append([0.,3.700 ,2.75 ,0.,0.,3.1])    #G8III
#        guess_set.append([0.,3.663 ,2.52 ,0.,0.,3.1])    #K1III
#        guess_set.append([0.,3.602 ,1.25 ,0.,0.,3.1])    #K5III
#        guess_set.append([0.,3.591 ,1.10 ,0.,0.,3.1])    #M0III

        guess_set.append([0., 3.760, 4.00, 0., 0., 3.1])    # horizontal branch
        guess_set.append([0., 3.720, 3.80, 0., 0., 3.1])    #
        guess_set.append([0., 3.680, 3.00, 0., 0., 3.1])    #
        guess_set.append([0., 3.700, 2.75, 0., 0., 3.1])    # G8III
        guess_set.append([0., 3.680, 2.45, 0., 0., 3.1])    #
        guess_set.append([0., 3.600, 1.20, 0., 0., 3.1])    # K5III
        guess_set.append([0., 3.580, 0.30, 0., 0., 3.1])    # K5III

        guess_set = np.array(guess_set)

        iso_objs = self.isochrones.query(guess_set[:, 0], guess_set[:, 1],
                                         guess_set[:, 2])

        guess_set[:, 4] = (
            np.log(np.maximum(0.007,
                              ((self.mag[self.init_bands[0]]
                                - self.mag[self.init_bands[1]])
                               - (iso_objs.abs_mag[self.init_bands[0]]
                                  - iso_objs.abs_mag[self.init_bands[1]])
                               )
                              / (iso_objs.AX1[self.init_bands[0]]
                                 [np.arange(guess_set.shape[0]), 11]
                                 - iso_objs.AX1[self.init_bands[1]]
                                 [np.arange(guess_set.shape[0]), 11]))))

        guess_set[:, 3] = (self.mag[self.init_bands[0]]
                           - (iso_objs.AX1[self.init_bands[0]]
                              [np.arange(guess_set.shape[0]), 11]
                              * guess_set[:, 4]
                              + iso_objs.AX2[self.init_bands[0]]
                              [np.arange(guess_set.shape[0]), 11]
                              * guess_set[:, 4] * guess_set[:, 4]
                              + iso_objs.abs_mag[self.init_bands[0]]))

        metal_min = sorted(self.isochrones.metal_dict.keys())[0]
        metal_max = sorted(self.isochrones.metal_dict.keys())[-1]

        for it in range(N_walkers):
            self.start_params[it, :] = (
                   guess_set[int(np.random.uniform()*len(guess_set))])
            self.start_params[it, 0] = (
                  metal_min+(metal_max-metal_min)*np.random.uniform())
            self.start_params[it, 5] = 2.9+0.4*np.random.uniform()

        self.Teff_chain = np.zeros(chain_length)
        self.logg_chain = np.zeros(chain_length)
        self.feh_chain = np.zeros(chain_length)
        self.dist_mod_chain = np.zeros(chain_length)
        self.logA_chain = np.zeros(chain_length)
        self.RV_chain = np.zeros(chain_length)

        self.prob_chain = np.zeros(chain_length)
        self.prior_chain = np.zeros(chain_length)
        self.Jac_chain = np.zeros(chain_length)
        self.accept_chain = np.zeros(chain_length)

        self.photom_chain = {}
        for band in self.mag:
            self.photom_chain[band] = np.zeros(chain_length)

        self.itnum_chain = np.zeros(chain_length)

    def emcee_run(self, iterations=10000, thin=10, burn_in=2000,
                  N_walkers=50, prune=True, prune_plot=False,
                  verbose_chain=True):
        """ emcee_run(iterations=10000, thin=10, burn_in=2000,
                      N_walkers=50, prune=True, prune_plot=False,
                      verbose_chain=True)

            Runs the emcee based inference of the posterior.

            Parameters
            ----------
            iterations : int, optional
                The number of iterations of emcee that will be
                performed.
            thin : int, optional
                A thinning factor that results in only 1 in
                every *thin* iterations being stored to the chain.
            burn_in : int, optinal
                Sets the length of the burn-in, during which
                nothing is retained to the chain.
            N_walkers : int, optinal
                The number of walkers to be used.
            prune : bool, optional
                Determines whether a pruning of obviously
                'lost' walkers is performed at the end of burn-in.
                These walkers are then dropped back onto
                randomly chosen 'good' walkers.
            prune_plot : bool, optional
                Produce a diagnostic plot showing what has
                happened during the pruning.
            verbose_chain : bool, optional
                Provides the option to store the state of a
                greater variety of parameters in the chain.
        """

        self.verbose_chain = verbose_chain

        self.emcee_init(N_walkers,
                        int((iterations-burn_in)/thin*N_walkers))

        sampler = emcee.EnsembleSampler(N_walkers, 6, emcee_prob, args=[self],
                                        a=1.5)

        # Burn-in
        pos, last_prob, state = sampler.run_mcmc(self.start_params, burn_in)
        sampler.reset()

        if prune:
            dbscan = sk_c.DBSCAN(eps=0.05)

            # pruning set
            pos, last_prob, state = sampler.run_mcmc(pos, 100, rstate0=state,
                                                     lnprob0=last_prob)
            dbscan.fit(sampler.flatchain[:, 1:2])
            labels = dbscan.labels_.astype(np.int)

            if prune_plot and mpl_present:

                fig = plt.figure()
                ax1 = fig.add_subplot(221)

                ax1.scatter(sampler.flatchain[:, 1], sampler.flatchain[:, 2],
                            color='0.5', s=1)
                ax1.scatter(pos[:, 1], pos[:, 2],
                            color=self.colors[labels].tolist(), s=3)
                ax1.set_xlim(right=3.5, left=4.5)
                ax1.set_ylim(bottom=5., top=2.)

                ax1 = fig.add_subplot(223)
                ax1.scatter(sampler.flatchain[:, 1], sampler.flatlnprobability,
                            color='0.5', s=1)
                ax1.scatter(pos[:, 1], last_prob,
                            color=self.colors[labels].tolist(), s=3)
                ax1.set_xlim(right=3.5, left=4.5)

            median_ln_prob = np.median(sampler.flatlnprobability)
            cl_list = []
            weights_list = []
            weights_sum = 0
            for cl_it in range(np.max(labels)+1):
                cl_list.append(posterior_cluster(
                            sampler.flatchain[labels == cl_it, :],
                            sampler.flatlnprobability[labels == cl_it]
                            - median_ln_prob))
                weights_sum += cl_list[-1].weight
                weights_list.append(cl_list[-1].weight)

            for i in range(N_walkers):
                cluster = np.random.choice(np.max(labels)+1,
                                           p=weights_list/np.sum(weights_list))
                index = int(np.random.uniform()
                            * len(cl_list[cluster]))
                pos[i, :] = cl_list[cluster].data[index, :]

            if prune_plot and mpl_present:
                ax1 = fig.add_subplot(222)

                ax1.scatter(sampler.flatchain[:, 1],
                            sampler.flatchain[:, 2], color='0.5', s=1)
                ax1.scatter(pos[:, 1], pos[:, 2],
                            color=self.colors[labels].tolist(), s=3)
                ax1.set_xlim(right=3.5, left=4.5)
                ax1.set_ylim(bottom=5., top=2.)

                ax1 = fig.add_subplot(224)
                ax1.scatter(sampler.flatchain[:, 1],
                            sampler.flatlnprobability, color='0.5',
                            s=1)
                ax1.scatter(pos[:, 1], last_prob,
                            color=self.colors[labels].tolist(), s=3)
                ax1.set_xlim(right=3.5, left=4.5)

                plt.tight_layout(pad=0.2, w_pad=0.1, h_pad=0.6)
                plt.savefig("prune.pdf")

            sampler.reset()


        if self.verbose_chain:

            # proper run
            for i, (pos, prob, rstate) in enumerate(
                  sampler.sample(pos, iterations=(iterations-burn_in),
                                 storechain=False)):

                if i % thin == 0:

                    start = int(i/thin*N_walkers)
                    end = int((i/thin+1)*N_walkers)

                    self.feh_chain[start:end] = pos[:, 0]
                    self.Teff_chain[start:end] = pos[:, 1]
                    self.logg_chain[start:end] = pos[:, 2]
                    self.dist_mod_chain[start:end] = pos[:, 3]
                    self.logA_chain[start:end] = pos[:, 4]
                    self.RV_chain[start:end] = pos[:, 5]

                    self.prob_chain[start:end] = prob

                    self.itnum_chain[start:end] = i

                    self.accept_chain[start:end] = sampler.acceptance_fraction

                    iso_obj = self.isochrones.query(pos[:, 0],
                                                    pos[:, 1],
                                                    pos[:, 2])
#                    A=np.exp(pos[:,4])

                    for band in self.mag:
                        self.photom_chain[band][start:end] = (
                                                iso_obj.abs_mag[band])

        else:
            pos, last_prob, state = sampler.run_mcmc(pos, iterations-burn_in,
                                                     thin=thin)

            self.feh_chain = sampler.flatchain[:, 0]
            self.Teff_chain = sampler.flatchain[:, 1]
            self.logg_chain = sampler.flatchain[:, 2]
            self.dist_mod_chain = sampler.flatchain[:, 3]
            self.logA_chain = sampler.flatchain[:, 4]
            self.RV_chain = sampler.flatchain[:, 5]

            self.prob_chain = sampler.flatlnprobability

        self.MCMC_run = True

    # ==============================================================
    # Functions to work with emcee sampler
    # EnsembleSampler version

    def emcee_ES_init(self, N_temps, N_walkers, chain_length):
        """ emcee_ES_init(N_walkers, chain_length)

            Initialises the emcee walkers for the ensemble
            sampler.

            Parameters
            ----------
            N_walkers : int
                The number of walkers to be used.
            chain_length: int
                The length of the MCMC chain
        """

        self.start_params = np.zeros([N_temps, N_walkers, 6])

        guess_set = []
        guess_set.append([0., 3.663, 4.57, 0., 0., 3.1])    # K4V
        guess_set.append([0., 3.672, 4.56, 0., 0., 3.1])    # K3V
        guess_set.append([0., 3.686, 4.55, 0., 0., 3.1])    # K2V
        guess_set.append([0., 3.695, 4.55, 0., 0., 3.1])    # K1V
        guess_set.append([0., 3.703, 4.57, 0., 0., 3.1])    # K0V
        guess_set.append([0., 3.720, 4.55, 0., 0., 3.1])    # G8V
        guess_set.append([0., 3.740, 4.49, 0., 0., 3.1])    # G5V
        guess_set.append([0., 3.763, 4.40, 0., 0., 3.1])    # G2V
        guess_set.append([0., 3.774, 4.39, 0., 0., 3.1])    # G0V
        guess_set.append([0., 3.789, 4.35, 0., 0., 3.1])    # F8V
        guess_set.append([0., 3.813, 4.34, 0., 0., 3.1])    # F5V
        guess_set.append([0., 3.845, 4.26, 0., 0., 3.1])    # F2V
        guess_set.append([0., 3.863, 4.28, 0., 0., 3.1])    # F0V
        guess_set.append([0., 3.903, 4.26, 0., 0., 3.1])    # A7V
        guess_set.append([0., 3.924, 4.22, 0., 0., 3.1])    # A5V
        guess_set.append([0., 3.949, 4.20, 0., 0., 3.1])    # A3V
        guess_set.append([0., 3.961, 4.16, 0., 0., 3.1])    # A2V

#        guess_set.append([0.,3.763 ,3.20 ,0.,0.,3.1])    #G2III
#        guess_set.append([0.,3.700 ,2.75 ,0.,0.,3.1])    #G8III
#        guess_set.append([0.,3.663 ,2.52 ,0.,0.,3.1])    #K1III
#        guess_set.append([0.,3.602 ,1.25 ,0.,0.,3.1])    #K5III
#        guess_set.append([0.,3.591 ,1.10 ,0.,0.,3.1])    #M0III

        guess_set.append([0., 3.760, 4.00, 0., 0., 3.1])    # horizontal branch
        guess_set.append([0., 3.720, 3.80, 0., 0., 3.1])    #
        guess_set.append([0., 3.680, 3.00, 0., 0., 3.1])    #
        guess_set.append([0., 3.700, 2.75, 0., 0., 3.1])    # G8III
        guess_set.append([0., 3.680, 2.45, 0., 0., 3.1])    #
        guess_set.append([0., 3.600, 1.20, 0., 0., 3.1])    # K5III
        guess_set.append([0., 3.580, 0.30, 0., 0., 3.1])    # K5III

        guess_set = np.array(guess_set)

        iso_objs = self.isochrones.query(guess_set[:, 0], guess_set[:, 1],
                                         guess_set[:, 2])

        guess_set[:, 4] = (
            np.log(((self.mag[self.init_bands[0]]
                     - self.mag[self.init_bands[1]])
                    - (iso_objs.abs_mag[self.init_bands[0]]
                       - iso_objs.abs_mag[self.init_bands[1]])
                    ) / (iso_objs.AX1[self.init_bands[0]]
                         [np.arange(guess_set.shape[0]), 11]
                         - iso_objs.AX1[self.init_bands[1]]
                         [np.arange(guess_set.shape[0]), 11])))

        guess_set[:, 3] = (self.mag[self.init_bands[0]]
                           - (iso_objs.AX1[self.init_bands[0]]
                              [np.arange(guess_set.shape[0]), 11]
                              * guess_set[:, 4]
                              + iso_objs.AX2[self.init_bands[0]]
                              [np.arange(guess_set.shape[0]), 11]
                              * guess_set[:, 4] * guess_set[:, 4]
                              + iso_objs.abs_mag[self.init_bands[0]]))

        metal_min = sorted(self.isochrones.metal_dict.keys())[0]
        metal_max = sorted(self.isochrones.metal_dict.keys())[-1]

        for it1 in range(N_temps):
            for it2 in range(N_walkers):
                self.start_params[it1, it2, :] = (
                    guess_set[int(np.random.uniform()*len(guess_set))])
                self.start_params[it1, it2, 0] = (
                    metal_min + (metal_max-metal_min) * np.random.uniform())
                self.start_params[it1, it2, 5] = (
                        2.9 + 0.4*np.random.uniform())

        self.Teff_chain = np.zeros(chain_length)
        self.logg_chain = np.zeros(chain_length)
        self.feh_chain = np.zeros(chain_length)
        self.dist_mod_chain = np.zeros(chain_length)
        self.logA_chain = np.zeros(chain_length)
        self.RV_chain = np.zeros(chain_length)

        self.prob_chain = np.zeros(chain_length)
        self.prior_chain = np.zeros(chain_length)
        self.Jac_chain = np.zeros(chain_length)
        self.accept_chain = np.zeros(chain_length)

        self.photom_chain = {}
        for band in self.mag:
            self.photom_chain[band] = np.zeros(chain_length)

        self.itnum_chain = np.zeros(chain_length)

    def emcee_ES_run(self, iterations=10000, thin=10, burn_in=2000,
                     N_temps=4, N_walkers=12, prune=True,
                     prune_plot=False, verbose_chain=True):
        """ emcee_run(iterations=10000, thin=10, burn_in=2000,
                      N_walkers=50, prune=True, prune_plot=False,
                      verbose_chain=True)

            Runs the emcee based inference of the posterior using
            the ensemble sampler.

            Parameters
            ----------
            iterations : int, optional
                The number of iterations of emcee that will be
                performed.
            thin : int, optional
                A thinning factor that results in only 1 in
                every *thin* iterations being stored to the chain.
            burn_in : int, optinal
                Sets the length of the burn-in, during which
                nothing is retained to the chain.
            N_walkers : int, optinal
                The number of walkers to be used.
            prune : bool, optional
                Determines whether a pruning of obviously
                'lost' walkers is performed at the end of burn-in.
                These walkers are then dropped back onto
                randomly chosen 'good' walkers.
            prune_plot : bool, optional
                Produce a diagnostic plot showing what has
                happened during the pruning.
            verbose_chain : bool, optional
                Provides the option to store the state of a
                greater variety of parameters in the chain.

        """
        self.verbose_chain = verbose_chain

        self.emcee_ES_init(N_temps, N_walkers,
                           int((iterations-burn_in)/thin*N_walkers))

        sampler = emcee.PTSampler(N_temps, N_walkers, 6, emcee_prob,
                                  lambda(x): np.zeros(x.shape[1]),
                                  loglargs=[self], a=1.5)

        # Burn-in
        pos, last_prob, state = sampler.run_mcmc(self.start_params,
                                                 burn_in)
        sampler.reset()

        if prune:
            dbscan = sk_c.DBSCAN(eps=0.05)

            # pruning set
            pos, last_prob, state = sampler.run_mcmc(pos, 100,
                                                     rstate0=state,
                                                     lnprob0=last_prob)
            dbscan.fit(sampler.flatchain[:, 1:2])
            labels = dbscan.labels_.astype(np.int)

            if prune_plot and mpl_present:

                fig = plt.figure()
                ax1 = fig.add_subplot(221)

                ax1.scatter(sampler.flatchain[:, 1],
                            sampler.flatchain[:, 2], color='0.5', s=1)
                ax1.scatter(pos[:, 1], pos[:, 2],
                            color=self.colors[labels].tolist(), s=3)
                ax1.set_xlim(right=3.5, left=4.5)
                ax1.set_ylim(bottom=5., top=2.)

                ax1 = fig.add_subplot(223)
                ax1.scatter(sampler.flatchain[:, 1],
                            sampler.flatlnprobability, color='0.5',
                            s=1)
                ax1.scatter(pos[:, 1], last_prob,
                            color=self.colors[labels].tolist(), s=3)
                ax1.set_xlim(right=3.5, left=4.5)

            median_ln_prob = np.median(sampler.flatlnprobability)
            cl_list = []
            weights_list = []
            weights_sum = 0
            for cl_it in range(np.max(labels)+1):
                cl_list.append(posterior_cluster(
                            sampler.flatchain[labels == cl_it, :],
                            sampler.flatlnprobability[labels == cl_it]
                            - median_ln_prob))
                weights_sum += cl_list[-1].weight
                weights_list.append(cl_list[-1].weight)

            for i in range(N_walkers):
                cluster = np.random.choice(np.max(labels)+1,
                                           p=weights_list/np.sum(weights_list))
                index = int(np.random.uniform()*len(cl_list[cluster]))
                pos[i, :] = cl_list[cluster].data[index, :]

            if prune_plot and mpl_present:
                ax1 = fig.add_subplot(222)

                ax1.scatter(sampler.flatchain[:, 1],
                            sampler.flatchain[:, 2], color='0.5', s=1)
                ax1.scatter(pos[:, 1], pos[:, 2],
                            color=self.colors[labels].tolist(), s=3)
                ax1.set_xlim(right=3.5, left=4.5)
                ax1.set_ylim(bottom=5., top=2.)

                ax1 = fig.add_subplot(224)
                ax1.scatter(sampler.flatchain[:, 1],
                            sampler.flatlnprobability, color='0.5',
                            s=1)
                ax1.scatter(pos[:, 1], last_prob,
                            color=self.colors[labels].tolist(), s=3)
                ax1.set_xlim(right=3.5, left=4.5)

                plt.tight_layout(pad=0.2, w_pad=0.1, h_pad=0.6)
                plt.savefig("prune.pdf")

            sampler.reset()

        if self.verbose_chain:

            for i, (pos, prob, rstate) in enumerate(
                  sampler.sample(pos, iterations=(iterations-burn_in),
                                 storechain=False)):      # proper run

                if i % thin == 0:

                    start = int(i/thin*N_walkers)
                    end = int((i/thin+1)*N_walkers)

                    self.feh_chain[start:end] = pos[0, :, 0]
                    self.Teff_chain[start:end] = pos[0, :, 1]
                    self.logg_chain[start:end] = pos[0, :, 2]
                    self.dist_mod_chain[start:end] = pos[0, :, 3]
                    self.logA_chain[start:end] = pos[0, :, 4]
                    self.RV_chain[start:end] = pos[0, :, 5]

                    self.prob_chain[start:end] = prob[0, :]

                    self.itnum_chain[start:end] = i

                    self.accept_chain[start:end] = (
                                sampler.acceptance_fraction[0, :])

                    iso_obj = self.isochrones.query(pos[0, :, 0],
                                                    pos[0, :, 1],
                                                    pos[0, :, 2])
#                    A=np.exp(pos[:,4])

                    for band in self.mag:
                        self.photom_chain[band][start:end] = (
                                            iso_obj.abs_mag[band])

        else:
            pos, last_prob, state = sampler.run_mcmc(pos, iterations-burn_in,
                                                     thin=thin)

            self.feh_chain = sampler.flatchain[:, 0]
            self.Teff_chain = sampler.flatchain[:, 1]
            self.logg_chain = sampler.flatchain[:, 2]
            self.dist_mod_chain = sampler.flatchain[:, 3]
            self.logA_chain = sampler.flatchain[:, 4]
            self.RV_chain = sampler.flatchain[:, 5]

            self.prob_chain = sampler.flatlnprobability

        self.MCMC_run = True

    # ==============================================================
    # Fit Gaussians

    def gmm_fit(self, max_components=10, verbose=False):

        """ gmm_fit(max_components=10)

            Fit a Gaussian mixture model to the (marginalised)
            MCMC chain in (disance_moduls, ln extinction,
            extinction law) space.

            Parameters
            __________
            max_components, int, optional
                The maximum size of the GMM (in terms of
                number of components) that will be fit.
            verbose : bool, optional
                Controls the verbosity of the function

            Notes
            _____
            Uses the Bayes Information Criterion (BIC) to
            select a number of componets, looking for a
            good fit, whilst peanalising models with more
            parameters.
        """

        if self.MCMC_run:
            fit_points = np.array([self.dist_mod_chain,
                                   self.logA_chain, self.RV_chain]).T

            # clean out any NaNs, infs, etc
            fit_points = fit_points[np.all(np.isfinite(fit_points), axis=1)]

            best_bic = +np.infty
            for n_components in range(1, max_components+1):
                gmm = sk_m.GaussianMixture(n_components=n_components,
                                           covariance_type='full',
                                           reg_covar=0.0001)
                gmm.fit(fit_points)
                if gmm.bic(fit_points) < best_bic-10:
                    best_bic = gmm.bic(fit_points)
                    self.best_gmm = gmm
                    if verbose:
                        print(n_components, best_bic, np.sort(gmm.weights_),
                              "*")
                else:
                    if verbose:
                        print(n_components, gmm.bic(fit_points),
                              np.sort(gmm.weights_))

    def gmm_sample(self, filename=None, num_samples=None):
        """ gmm_sample(filename=None, num_samples=None)

            Sample from the Gaussian mixture model that has
            been fit to the data.

            Parameters
            ----------
            filename : string, optional
                A file to which the samples will be written
            num_samples : int, optional
                The number of samples tha will be drawn. If
                None, the number of samples matches the length
                of the MCMC chain.
        """

        if self.best_gmm:
            if num_samples is None:
                num_samples = self.prob_chain.size

            components = np.random.choice(self.best_gmm.weights_.size,
                                          p=self.best_gmm.weights_,
                                          size=num_samples)

            covar_roots = []
            for covar in self.best_gmm._get_covars():
                covar_roots.append(np.linalg.cholesky(covar))

            self.gmm_sample = self.best_gmm.means_[components]
            for it in range(components.size):
                self.gmm_sample[it, :] += np.dot(
                            covar_roots[components[it]],
                            np.random.normal(size=(2, 1))).flatten()

            if filename:
                np.savetxt(filename, self.gmm_sample)

    # ==============================================================
    # Auxilary functions

    def plot_MCMCsample(self):
        """ plot_MCMCsample()

            Plot the MCMC sample on the ln(s) ln(A) plane
            on the screen.
        """

        # Raise an error if matplotlib not available
        if not mpl_present:
            raise ImportError

        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        ax1.scatter(self.dist_mod_chain, self.logA_chain, marker='.')
        plt.show()

    def chain_dump(self, filename):
        """ chain_dum(filename)

            Dump the MCMC chain to a text file

            Parameters
            ----------
            filename : string
                The file to which the chain will be written.
        """

        X = [self.itnum_chain, self.Teff_chain, self.logg_chain,
             self.feh_chain, self.dist_mod_chain, self.logA_chain,
             self.RV_chain, self.prob_chain, self.prior_chain,
             self.Jac_chain, self.accept_chain]
        header_txt = ("N\tTeff\tlogg\tfeh\tdist_mod\tlogA\tRV\tlike\t"
                      "prior\tJac\taccept")
        for band in self.photom_chain:
            X.append(self.photom_chain[band])
            header_txt += "\t{}".format(band)
        X = np.array(X).T
        header_txt += "\n"

        np.savetxt(filename, X, header=header_txt)

    def plot_MCMCsample_gaussians(self):
        """plot_MCMCsample_gaussians()

            Plot MCMC sample overlaid with gaussian fit
            in (distance modulus, ln extinction) space
            to the screen.
        """

        # Raise an error if matplotlib not available
        if not mpl_present:
            raise ImportError

        fit_points = np.array([self.dist_mod_chain,
                               self.logA_chain]).T
        Y_ = self.best_gmm.predict(fit_points)

        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        for it in range(self.best_gmm.weights_.size):
            ax1.scatter(fit_points[Y_ == it, 0], fit_points[Y_ == it, 1],
                        marker='.', color=self.colors[it])

            # Plot an ellipse to show the Gaussian component

            v, w = linalg.eigh(self.best_gmm._get_covars()[it])

            angle = np.arctan2(w[0][1], w[0][0])
            angle = 180 * angle / np.pi  # convert to degrees
            v *= 4
            ell = mpl.patches.Ellipse(self.best_gmm.means_[it], v[0],
                                      v[1], 180 + angle,
                                      ec=self.colors[it], fc='none',
                                      lw=3)
            ell.set_clip_box(ax1.bbox)
            ell.set_alpha(.5)
            ax1.add_artist(ell)
        plt.show()

    def compare_MCMC_hist(self):
        """ compare_MCMC_hist()

            Produce a plot that compares the GMM
            approximation to the estimated posterior,
            showing the contribution of each component.
        """

        # Raise an error if matplotlib not available
        if not mpl_present:
            raise ImportError

        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        bins = np.arange(8., 17.5, 0.25)

        ax1.hist(self.dist_mod_chain, bins, histtype='step', ec='k')

        x = np.arange(np.min(self.dist_mod_chain),
                      np.max(self.dist_mod_chain), 0.1)
        y = np.zeros(x.size)

        for it in range(self.best_gmm.weights_.size):
            y += (1/np.sqrt(2*np.pi*self.best_gmm._get_covars()[it][0, 0])
                  * np.exp(-np.power(x-self.best_gmm.means_[it][0], 2)
                  / (2*self.best_gmm._get_covars()[it][0, 0]))
                  * self.best_gmm.weights_[it])
            y_it = (1/np.sqrt(2*np.pi*self.best_gmm._get_covars()[it][0, 0])
                    * np.exp(-np.power(x-self.best_gmm.means_[it][0], 2)
                    / (2*self.best_gmm._get_covars()[it][0, 0]))
                    * self.dist_mod_chain.size*.25
                    * self.best_gmm.weights_[it])
            ax1.plot(x, y_it, color=self.colors[it])

        y *= self.dist_mod_chain.size*.25
        ax1.plot(x, y, 'k--', linewidth=1.5)

        plt.show()


class posterior_cluster:
    """ A class to store clusters in posterior space
    """

    def __init__(self, data, probs):
        """ __init__(data, probs)

            Initialise a cluster in posterior space.

            Parameters
            ----------
            data : ndarray(float)
                The coordinates of the data points associated
                with the cluster
            probs : ndarray(float)
                The probabilities of each of the data points
        """

        self.data = data
        self.probs = probs

        self.set_weight()

    def __len__(self):
        """ __len__()

            Gives the number of points in the cluster

            Returns
            -------
            The number of points in the cluster
        """
        return self.data.shape[0]

    def set_weight(self, weight=None):
        """ set_weight(weight=None)

            Sets the probability weight of the cluster. If no
            weight is provided, the weight is set to the mean
            of the probabilities of each point in the cluster
            multiplied by the standard deviation of the cluster
            member positions (with a floor).

            Parameters
            ----------
            weight : float
                The probaility weight of the cluster
        """

        if weight:
            self.weight = weight
        else:
            self.weight = (np.mean(np.exp(self.probs))
                           * max(np.std(self.data[:, 1]), 0.01)
                           * max(np.std(self.data[:, 2]), 0.01))
