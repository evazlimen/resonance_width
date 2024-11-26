"""
integration_functions.py
Script containing the numerical integration functions

**Note that we are encoding [phi, n, e, pomega, lambda] as the array eqns**
"""

#imports
from scipy.integrate import solve_ivp


def rhs(t, eqns, j, beta, mu_prime, tau_n, tau_e, p, nprime):
    """
    Defines the differential equations .

    Arguments:
        eqns :  vector of the state variables:
                eqns = [phi, n, e, pomega]
        t :  time
        j: order of resonance (j: (j+1))
        beta: defined as 0.8*j. from the disturbing function
        mu_prime: ratio of outer planet mass to star mass
        tau_n: migration timescale
        tau_e: damping timescale
        p: accounts for the contribution of eccentricity damping to changing the mean motion. 3 is used for this paper
        nprime: mean motion of the outer planet
    """
    consts = [j, beta, mu_prime, tau_n, tau_e, p, nprime]

    pomegadot = -consts[1]*consts[2]*eqns[1]*np.cos(eqns[0])/eqns[2] #pomega is longitude of pericenter

    phidot= (consts[0]+1)*nprime -consts[0]*eqns[1] - pomegadot #φ is a measure of the displacement of the longitude of conjunction from the inner planet’s pericenter.

    ndot = 3*consts[0]*consts[1]*consts[2]*eqns[2]*eqns[1]**2*np.sin(eqns[0])-eqns[1]/consts[3]+(consts[5]*eqns[2]**2*eqns[1]/consts[4]) #n is mean motion of pericenter.

    edot = consts[1]*consts[2]*eqns[1]*np.sin(eqns[0])-eqns[2]/consts[4] #eccentricity

    lambdadot = eqns[1]

    holder = np.zeros(5) #just doing this so everything is a float
    holder[0] = phidot
    holder[1] = ndot
    holder[2] = edot
    holder[3] = pomegadot
    holder[4] = lambdadot

    return holder

def integration_time(t_initial,t_final, stepsize):
    """
    Function to create an array of values over which to output values from the numerical integration

    Parameters
    ----------
    t_initial : float
        Initial time to integrate from
    t_final : float
        Final timepoint after which integration is complete
    stepsize : float
        size of the space between timepoints

    Returns
    -------
    t_span 
        Array of the initial and final timepoint
    t_eval
        Array of timepoints for which solutions to the numerical integration will be outputted
    """
    t_span = (t_initial, t_final)
    number = int(t_final/stepsize)
    #where it is evaluated
    t_eval=np.linspace(t_initial, t_final, num=number)
    return (t_span, t_eval)


def integrate(rhs,t_span,initials,t_eval,j,beta,mu_prime,tau_n,tau_e,p,nprime, rtol_param):
    """
    Function to numerically integrate rhs()

    Parameters
    ----------
    rhs : array-like
        array of coupled differential equations
    t_span : array
        array of [t_initial, t_final]
    initials : array
        array of user-specified initial conditions. initials=  [phi_initial, n_initial, e_initial, pomega_initial, lambda_initial]
    t_eval : array
        Array of timepoints for which solutions to the numerical integration will be outputted
    j : integer
        the order of the resonance
    beta : float
        the disturbing function
    mu_prime : float
        the ratio of the outer planet mass to the star mass
    tau_n : float
        Migration timescale (years)
    tau_e : float
        eccentricity damping timescale (years)
    p : integer
        accounts for the contribution of eccentricity damping to changing the mean motion. 3 is used for this paper
    nprime : float
        initial mean motion of the inner planet
    rtol_param : float
        atol sets number of correct decimal places
        To achieve the desired atol set rtol such that rtol * abs(y) is always smaller than atol.
        In doubt, use 1e-6.
    Returns
    -------
    res
        Array of timepoints and solutions for each equation of the coupled differential equations
    """
    #evaluate
    #atol sets number of correct decimal places
    """To achieve the desired atol set rtol such that rtol * abs(y) is always smaller than atol."""
    res = solve_ivp(rhs, t_span, initials, method='RK45', t_eval=t_eval, dense_output = True, max_step = 1, rtol = rtol_param, args=(j, beta, mu_prime, tau_n, tau_e, p, nprime)) #need integrator to capture small oscillations, not smooth
    print(res.status)
    return res