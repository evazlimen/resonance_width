"""
plotting_functions.py
Script containing the plotting functions to plot results of numerical integration
"""

#imports:

#used in colored_line():
from matplotlib.collections import LineCollection
import warnings


#used in plot_runs():
import matplotlib.pyplot as plt
plt.style.use("/Library/Frameworks/Python.framework/versions/3.13/lib/python3.13/site-packages/matplotlib/style/my_style.mplstyle")


def colored_line(x, y, c, ax, **lc_kwargs):
    """
    This is just so that our resonance curve plots look cool
    This was taken from stack exchange
    Plot a line with a color specified along the line by a third value.

    It does this by creating a collection of line segments. Each line segment is
    made up of two straight lines each connecting the current (x, y) point to the
    midpoints of the lines connecting the current point with its two neighbors.
    This creates a smooth line with no gaps between the line segments.

    Parameters
    ----------
    x, y : array-like
        The horizontal and vertical coordinates of the data points.
    c : array-like
        The color values, which should be the same size as x and y.
    ax : Axes
        Axis object on which to plot the colored line.
    **lc_kwargs
        Any additional arguments to pass to matplotlib.collections.LineCollection
        constructor. This should not include the array keyword argument because
        that is set to the color argument. If provided, it will be overridden.

    Returns
    -------
    matplotlib.collections.LineCollection
        The generated line collection representing the colored line.
    """
    if "array" in lc_kwargs:
        warnings.warn('The provided "array" keyword argument will be overridden')

    # Default the capstyle to butt so that the line segments smoothly line up
    default_kwargs = {"capstyle": "butt"}
    default_kwargs.update(lc_kwargs)

    # Compute the midpoints of the line segments. Include the first and last points
    # twice so we don't need any special syntax later to handle them.
    x = np.asarray(x)
    y = np.asarray(y)
    x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))

    # Determine the start, middle, and end coordinate pair of each line segment.
    # Use the reshape to add an extra dimension so each pair of points is in its
    # own list. Then concatenate them to create:
    # [
    #   [(x1_start, y1_start), (x1_mid, y1_mid), (x1_end, y1_end)],
    #   [(x2_start, y2_start), (x2_mid, y2_mid), (x2_end, y2_end)],
    #   ...
    # ]
    coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
    coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
    coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

    lc = LineCollection(segments, **default_kwargs)
    lc.set_array(c)  # set the colors of each segment

    return ax.add_collection(lc)


def plot_runs(res, tau_n, n_initial, mu_prime,tau_rat, t_final, j, beta, t_eval, nprime, path):
    """
    Plot the results of our integrations with 3 plots: circular parameterization, period ratio, and resonant angle time evolution

    Parameters
    ----------
    res : array-like
        The x and y data points from the output of the integration
    tau_n : float
        Migration timescale (years)
    n_initial : float
        initial mean motion of the non-evolving outer planet
    mu_prime : float
        ratio of the outer planet mass to star mass
    tau_rat : float
        ratio of tau_e (eccentricity damping timescale) to tau_n (migration damping timescale)
    t_final : float
        final timepoint of the integration
    j : integer
        the order of the resonance
    beta : float
        the disturbing function
    t_eval : array
        array of time points that the numerical integration results are outputted at
    nprime : float
        initial mean motion of the inner planet
    path : string
        path of where to save the figure

    Returns
    -------
    matplotlib.figure
        Figure with 3 subfigures
    """
    tau_e = tau_rat*tau_n
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (14,4))
    fig.suptitle(r"$\tau_{n}$=%.1e, $n_{initial}$=%0.2f, $\mathbf{\mu'=%.6e}$, $\tau_{e}=$%.1e, $\tau_{e}/\tau_{n}$=%.5f" % (tau_n, n_initial, mu_prime, tau_rat*tau_n, tau_rat))
    nth=1

    color = np.linspace(0, t_final, res.t.size)
    lines = colored_line(res.y[2][::nth]*np.cos(res.y[0][::nth]), res.y[2][::nth]*np.sin(res.y[0][::nth]), color/(10**5), ax1, linewidth=1, cmap="plasma",alpha = .4)
    cbar = fig.colorbar(lines, label =r"x$10^{5}$")  # add a color legend


    e_eq = func_e_eq(tau_e, j, tau_n)

    sin_phi_eq = func_sin_phi_eq(e_eq,beta, mu_prime,tau_e,n_initial)
    ax1.scatter(e_eq*np.cos(np.arcsin(sin_phi_eq)), e_eq*sin_phi_eq, color = "black", s = 5, label = "equilibrum")
    ax1.set_xlabel(r"$e\: cos\phi$")
    ax1.set_ylabel(r"$e\: sin\phi$")


    ax2.plot(t_eval/(tau_n*nprime), res.y[1]/nprime, lw = .1) #P_2/P_1 = n/n'. 2 is the outer planet (not evolving)
    ax2.set_xlabel(r"x$10^{%d}$ (x $n'\tau_{n})$" %(int(np.log10(tau_n))))
    ax2.set_ylabel(r"$P_2/P_1$")
    #ax2.set_ylim(2.014, 2.016)
    #ax2.set_xlim(1, 1.05)

    """
    ax3.plot(res.t/(tau_n*nprime),np.abs(np.sin(res.y[0])))
    ax3.hlines((sin_phi_eq), 0, max(res.t/(10**5)), color = "black", label = r"$sin{\phi_{eq}}$")
    ax3.legend(loc = "upper right")
    ax3.set_xlabel(r"x$10^{%d}$ (x $n'\tau_{n})$" %(int(np.log10(tau_n))))
    ax3.set_ylabel(r"$|sin{\phi}|$")
    """
    fig.tight_layout()
    plt.savefig(path + str(tau_rat) + ".png")
