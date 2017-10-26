from glob import glob
import math
import matplotlib as mpl
import numpy as np
from os import environ
import sklearn.mixture as sk_m
from scipy import linalg
from tqdm import tqdm
import warnings

import marg_iso as mi
import isolib as il

environ['MKL_NUM_THREADS'] = '1'
environ['NUMEXPR_NUM_THREADS'] = '1'

warnings.filterwarnings("ignore", category=DeprecationWarning)

# =============================================================================

# import isochrones

isochrone_lib = il.iso_grid_tefflogg("padova_iphas-UKIDSS.txt",
                                     bands=['r_INT', 'i_INT', 'Ha_INT',
                                            'J_UKIDSS', 'H_UKIDSS',
                                            'K_UKIDSS'])


# =============================================================================
# Read in photometry

filenames = glob("input_catalogues/*.txt")

for filename in filenames:
    output_filename = "output_catalogues/{0}_out.txt".format(
                        filename.lstrip("input_catalogues/").rstrip(".txt"))
    print("output to: ", output_filename)
    output = open(output_filename, "w")

    photom_data = np.genfromtxt(filename,
                                usecols=(2, 3, 4, 5, 6, 7, 10, 11,
                                         17, 18, 19, 20, 21, 22))
    ll = photom_data[:, 6]
    bb = photom_data[:, 7]
    mags = []
    d_mags = []

    for line in photom_data:
        mags.append({'r_INT': line[0], 'i_INT': line[2], 'Ha_INT': line[4]})
        d_mags.append({'r_INT': line[1], 'i_INT': line[3], 'Ha_INT': line[5]})

        #if not (math.isnan(line[8]) or math.isnan(line[9])):
            #mags[-1]['J_UKIDSS'] = line[8]
            #d_mags[-1]['J_UKIDSS'] = line[9]
        #if not (math.isnan(line[10]) or math.isnan(line[11])):
            #mags[-1]['H_UKIDSS'] = line[10]
            #d_mags[-1]['H_UKIDSS'] = line[11]
        #if not (math.isnan(line[12]) or math.isnan(line[13])):
            #mags[-1]['K_UKIDSS'] = line[12]
            #d_mags[-1]['K_UKIDSS'] = line[13]

# =============================================================================
# Run MCMC etc

    for i in tqdm(range(ll.size)):

        if mags[i]['r_INT'] > 19:
            continue

        star1 = mi.star_posterior(ll[i], bb[i], mags[i], d_mags[i],
                                  isochrones=isochrone_lib,
                                  init_bands=["r_INT", "i_INT"])

        # There is a choice of samplers available
        # - comment out the unwanted one

        star1.emcee_run(thin=10, iterations=20000, prune_plot=False,
                         prune=True)
        #star1.emcee_ES_run(N_temps=8, thin=10, iterations=10000, burn_in=5000,
                           #prune_plot=False, prune=False)
        star1.gmm_fit(6)

        star1.chain_dump("chain.txt")

# =============================================================================
# Dump gmm params to file

        output.write("{0:.5f}\t{1:.5f}\t".format(ll[i], bb[i]))
        for it in range(star1.best_gmm.weights_.size):
            output.write("{0:.5G}\t{1:.5G}\t{2:.5G}\t{3:.5G}\t{4:.5G}\t"
                         "{5:.5G}\t{6:.5G}\t{7:.5G}\t{8:.5G}\t{9:.5G}\t"
                         .format(star1.best_gmm.weights_[it],
                                 star1.best_gmm.means_[it][0],
                                 star1.best_gmm.means_[it][1],
                                 star1.best_gmm.means_[it][2],
                                 star1.best_gmm.covariances_[it][0, 0],
                                 star1.best_gmm.covariances_[it][1, 0],
                                 star1.best_gmm.covariances_[it][1, 1],
                                 star1.best_gmm.covariances_[it][2, 0],
                                 star1.best_gmm.covariances_[it][2, 1],
                                 star1.best_gmm.covariances_[it][2, 2]))
        output.write("\n")

# =============================================================================
# Plot likelihoods/posteriors

        plot_filename = "output_plots/{0}_{1}.pdf".format(
                    filename.lstrip("input_catalogues/").rstrip(".txt"), i)

        mi.posterior_quadplot(star1, plot_filename)
