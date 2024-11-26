"""
helper_functions.py
Script containing the helper functions to run the numerical integration
"""

def func_muprime_min(j,beta,taue_taun):
    """
    Calculate mu' below which the planet will be permanently trapped in resonance
    Equation 26 from Goldreich & Schlichting 2014.

    Parameters
    ----------
    j : integer
        the order of the resonance
    beta : float
        the disturbing function
    taue_taun : float
        the ratio of tau_e to tau_n. (eccentricity damping timescale to migration timescale)
    Returns
    -------
    muprime_min
        the minimum mu' as calculated by equation 26
    """
    muprime_min = j/(np.sqrt(3)*(j+1)**(3/2)*beta)*(taue_taun)**(3/2)
    return muprime_min 

def func_muprime_max(j,beta,taue_taun):
    """
    Calculate mu' above which the planet will escape resonance
    Equation 28 from Goldreich & Schlichting 2014.

    Parameters
    ----------
    j : integer
        the order of the resonance
    beta : float
        the disturbing function
    taue_taun : float
        the ratio of tau_e to tau_n. (eccentricity damping timescale to migration timescale)
    Returns
    -------
    muprime_max
        the maximum mu' as calculated by equation 26
    """
    muprime_max = (j**2/(8*np.sqrt(3)*(j+1)**(3/2)*beta))*(taue_taun)**(3/2)
    return muprime_max


#equilibrium values
def func_e_eq(tau_e, j, tau_n):
    """
    Calculate the equilibrium eccentricity in resonance
    Equation 24 from Goldreich & Schlichting 2014.

    Parameters
    ----------
    tau_e : float
        eccentricity damping timescale (years)
    j : integer
        order of the resonance
    tau_n: float
        migration timescale (years)
    Returns
    -------
    e_eq
        the equilibrium eccentricity as calculated by equation 24
    """
    e_eq = (tau_e/(3*(j+1)*tau_n))**.5
    return e_eq

def func_sin_phi_eq(e_eq,beta, mu_prime,tau_e,n_initial):
    """
    Calculate sine of the equilbirum resonant angle
    Equation 25 from Goldreich & Schlichting 2014.

    Parameters
    ----------
    e_eq : float
        equilibrium eccentricity in resonance
    beta : float
        the disturbing function
    mu_prime : float
        the ratio of the outer planet mass to the star mass
    tau_e : float
        eccentricity damping timescale
    n_initial : float
        initial mean motion value set for the stationary outer planet
    Returns
    -------
    sin_phi_eq
        sine of the equilibrium value as calculated by equation 25
    """
    sin_phi_eq = e_eq/(beta*mu_prime*tau_e*n_initial)
    return sin_phi_eq


def func_tau_rat_vals(j,beta, mu_prime, number, number_outside):
    """
    Create an array of tau_e/tau_n ratios to use as inputs to integration

    Parameters
    ----------
    j : integer
        the order of the resonance
    beta : float
        the disturbing function
    mu_prime : float
        the ratio of the outer planet mass to the star mass
    number : integer
        number of ratios to create (number of integations that will be performed)
    number_outside: integer
        how many total ratios outside of the range from minimum tau ratio to maximum tau ratio that will be created. So of your
        number of ratios created set by number, they will range from a minimum ratio to a maximum ratio, plus additional ratio(s) less than the minimum and more than
        the maximum. If number_outside = 2, then there will be one ratio below the minimum and one above the maximum.
    Returns
    -------
    tau_rat_vals
        an array of tau_e/tau_n ratios to integrate
    """
    tau_ratio_1 = ((mu_prime*np.sqrt(3)*(j+1)**(3/2)*beta)/(j))**(2/3)
    tau_ratio_2 = ((mu_prime*8*np.sqrt(3)*(j+1)**(3/2)*beta)/(j**2))**(2/3)

    tau_span = tau_ratio_1-tau_ratio_2

    t_min = tau_ratio_1 - (number_outside*(tau_span)/number)
    t_max = tau_ratio_2 + (number_outside*(tau_span)/number)

    tau_rat_vals = np.linspace(t_min, t_max, num = number)
    
    return tau_rat_vals