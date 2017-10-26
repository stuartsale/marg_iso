import math
import numpy as np
from scipy import linalg
import sklearn.mixture as sk_m

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors
from mpl_toolkits.axes_grid1 import ImageGrid, Divider
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator

fig_width_pt = 240.0  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inch
golden_mean = (math.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = 2*fig_width_pt*inches_per_pt  # width in inches
fig_height = fig_width*golden_mean      # height in inches
fig_size = [fig_width, fig_height]
params = {'backend': 'ps', 'axes.labelsize': 10, 'text.fontsize': 10,
          'legend.fontsize': 10, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
          'text.usetex': True, 'figure.figsize': fig_size,
          'font.weight': "bold", 'ytick.major.pad': 5, 'xtick.major.pad': 5,
          'axes.titlesize': 10, 'ps.distiller.res': 24000}
mpl.rcParams.update(params)
mpl.rc("text.latex", preamble="\usepackage{bm}")

colors = np.array([x for x in 'bgrcmybgrcmybgrcmybgrcmy'])

# =============================================================================


def posterior_quadplot(star_obj, output_filename):
    """ posterior_quadplot(star_obj, output_file)

        Make a plot of the posterior distribution
        p(a, mu, R | data, Galaxy_model, flat_a_prior)

        Parameters
        ----------
        star_obj : star_posterior
            The object that contains the emcee chains and the
            GMM approximation to them
        output_filename : str
            The name of a file to which the plot will be saved

        Returns
        -------
        None
    """

    # Make figure object

    fig = plt.figure()
    fig.subplots_adjust(left=0.075, right=0.87, bottom=0.15, top=0.95,
                        wspace=0.4, hspace=0.4)

    # Setup subplots

    ax1 = fig.add_subplot(221)
    ax1.set_xlabel(r"$ \mu $")
    ax1.set_ylabel(r"$ a_{4000}$")

    ax2 = fig.add_subplot(222)
    ax2.set_xlabel(r"$ \mu $")
    ax2.set_ylabel(r"$ R_{5495}$")

    ax3 = fig.add_subplot(223)
    ax3.set_xlabel(r"$ \mu $")
    ax3.set_ylabel(r"$ a_{4000}$")

    ax4 = fig.add_subplot(224)
    ax4.set_xlabel(r"$ \mu $")
    ax4.set_ylabel(r"$ R_{5495}$")

    # =========================================================================
    # mu x a_4000

    plot_array = np.array([star_obj.dist_mod_chain, star_obj.logA_chain]).T
    plot_array = plot_array[np.all(np.isfinite(plot_array), axis=1)].T

    H, xedges, yedges = np.histogram2d(plot_array[0], plot_array[1], bins=50)
    X, Y = np.meshgrid(xedges, yedges)
    extent1 = [np.min(X), np.max(X), np.min(Y), np.max(Y)]
    ax1.imshow(H.T/np.max(H), origin="lower", extent=extent1,
               aspect=0.6*(extent1[1]-extent1[0])/(extent1[3]-extent1[2]),
               interpolation='nearest', cmap='Greys',
               norm=LogNorm(vmin=0.001, vmax=1.))
    ax1.xaxis.set_major_locator(MaxNLocator(6, integer=True))
    ax1.yaxis.set_major_locator(MaxNLocator(6))

    x_mid = (xedges[:-1]+xedges[1:]) / 2.
    y_mid = (yedges[:-1]+yedges[1:]) / 2.

    Xmid, Ymid = np.meshgrid(x_mid, y_mid)
    shape1 = Xmid.shape
    Xmid = np.ravel(Xmid)
    Ymid = np.ravel(Ymid)

    marg_gmm = sk_m.GMM(n_components=star_obj.best_gmm.weights_.size,
                        covariance_type='full')
    marg_gmm.weights_ = star_obj.best_gmm.weights_
    marg_gmm.means_ = star_obj.best_gmm.means_[:, :-1]
    marg_gmm.covars_ = star_obj.best_gmm.covariances_[:, :-1, :-1]

    score_array = marg_gmm.score(np.array([Xmid, Ymid]).T)

    score_array = np.exp(score_array.reshape(shape1))
    score_array = (score_array / np.sum(score_array)
                   * star_obj.dist_mod_chain.size)
    ax3.imshow(score_array/np.max(H), origin="lower", extent=extent1,
               aspect=0.6*(extent1[1]-extent1[0])/(extent1[3]-extent1[2]),
               interpolation='nearest', cmap='Greys',
               norm=LogNorm(vmin=0.001, vmax=1))
    ax3.xaxis.set_major_locator(MaxNLocator(6, integer=True))
    ax3.yaxis.set_major_locator(MaxNLocator(6))

    for it in range(star_obj.best_gmm.weights_.size):

        # Plot an ellipse to show the Gaussian component

        v, w = linalg.eigh(star_obj.best_gmm.covariances_[it][:2, :2])

        angle = np.arctan2(w[0][1], w[0][0])
        angle = 180 * angle / np.pi  # convert to degrees
        v *= 4
        ell = mpl.patches.Ellipse(star_obj.best_gmm.means_[it][:2], v[0], v[1],
                                  180 + angle, ec=colors[it], fc='none',
                                  lw=8*star_obj.best_gmm.weights_[it])
        ell.set_clip_box(ax3.bbox)
        ell.set_alpha(.5)
        ax3.add_artist(ell)

    # =========================================================================
    # mu x R

    plot_array = np.array([star_obj.dist_mod_chain, star_obj.RV_chain]).T
    plot_array = plot_array[np.all(np.isfinite(plot_array), axis=1)].T

    H, xedges, yedges = np.histogram2d(plot_array[0], plot_array[1], bins=50)
    X, Y = np.meshgrid(xedges, yedges)
    extent1 = [np.min(X), np.max(X), np.min(Y), np.max(Y)]
    im = ax2.imshow(H.T/np.max(H), origin="lower", extent=extent1,
                    aspect=0.6*(extent1[1]-extent1[0])/(extent1[3]-extent1[2]),
                    interpolation='nearest', cmap='Greys',
                    norm=LogNorm(vmin=0.001, vmax=1))
    ax2.xaxis.set_major_locator(MaxNLocator(6, integer=True))
    ax2.yaxis.set_major_locator(MaxNLocator(6))

    x_mid = (xedges[:-1]+xedges[1:]) / 2.
    y_mid = (yedges[:-1]+yedges[1:]) / 2.

    Xmid, Ymid = np.meshgrid(x_mid, y_mid)
    shape1 = Xmid.shape
    Xmid = np.ravel(Xmid)
    Ymid = np.ravel(Ymid)

    marg_gmm = sk_m.GMM(n_components=star_obj.best_gmm.weights_.size,
                        covariance_type='full')
    marg_gmm.weights_ = star_obj.best_gmm.weights_
    marg_gmm.means_ = star_obj.best_gmm.means_[:, ::2]
    marg_gmm.covars_ = star_obj.best_gmm.covariances_[:, ::2, ::2]

    score_array = marg_gmm.score(np.array([Xmid, Ymid]).T)

    score_array = np.exp(score_array.reshape(shape1))
    score_array = (score_array / np.sum(score_array)
                   * star_obj.dist_mod_chain.size)
    ax4.imshow(score_array/np.max(H), origin="lower", extent=extent1,
               aspect=0.6*(extent1[1]-extent1[0])/(extent1[3]-extent1[2]),
               interpolation='nearest', cmap='Greys',
               norm=LogNorm(vmin=0.001, vmax=1.))
    ax4.xaxis.set_major_locator(MaxNLocator(6, integer=True))
    ax4.yaxis.set_major_locator(MaxNLocator(6))

    for it in range(star_obj.best_gmm.weights_.size):

        # Plot an ellipse to show the Gaussian component

        v, w = linalg.eigh(star_obj.best_gmm.covariances_[it][::2, ::2])

        angle = np.arctan2(w[0][1], w[0][0])
        angle = 180 * angle / np.pi  # convert to degrees
        v *= 4
        ell = mpl.patches.Ellipse(star_obj.best_gmm.means_[it][::2], v[0],
                                  v[1], 180 + angle, ec=colors[it], fc='none',
                                  lw=8*star_obj.best_gmm.weights_[it])
        ell.set_clip_box(ax4.bbox)
        ell.set_alpha(.5)
        ax4.add_artist(ell)

    # =========================================================================
    # Colourbar

    cbar_ax = fig.add_axes([0.9, 0.1, 0.02, 0.8])
    H = np.arange(0., 1., 0.1)
    cbar1 = fig.colorbar(im, cax=cbar_ax)
    cbar1.set_label(r"${\rm count / max. count}$")
    cbar1.set_ticks([0.001, 0.01, 0.1, 1.])
    cbar1.set_ticklabels([r'$0.001$', r'$0.01$', r'$0.1$', r'$1$'])
    cbar1.solids.set_edgecolor("face")

    try:
        fig.savefig(output_filename)
    except OSError:
        print "Star figure failed to print".format(i)
