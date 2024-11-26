"""
Script to perform and plot the numerical integration
"""
#imports
import numpy as np
import helper_functions.py
import plotting_functions.py
import integration_functions.py

#constants that we literally never change
#define some initial conditions
n_initial = 1.05#somewhat offset from 1, becuase we are doing a 2:1 resoance and we have set nprime to be 1/2 (ie j=1, so (j+1)nprime = 2*nprime = 1 )
phi_initial = 0
e_initial = 0.01
pomega_initial = 0
lambda_initial = 0
#
initials=  [phi_initial, n_initial, e_initial, pomega_initial, lambda_initial]

p=3 #arbitrarily set but can be other values sometimes. Constant throughout integration.
j=1 #2:1 resonance
nprime= 0.5
beta = 0.8*j #disturbing function thing

#integration constants
t_initial = 0
t_final = 2e7
stepsize = 500
rtol_param = 1e-6 # parameter to control accuracy of solution in solve_ivp


#how many tau ratios are we doing:
number = 16
#how many steps on either side outside of the theory lines are we looking at:
number_outside = 3

#constants that we do change sometimes
tau_n=10**7/nprime #migration timescale
mu_prime = 3e-3 #ratio of outer planet mass to star mass


#get the ics for all the runs
tau_rat_vals = func_tau_rat_vals(j, beta, mu_prime, number, number_outside)
print(tau_rat_vals)

t_span = integration_time(t_initial,t_final, stepsize)[0]
t_eval = integration_time(t_initial,t_final, stepsize)[1]

#path to save figures to
path = "/Users/evazlimen/Documents/research/resonance_width/plots/"

# %%
for i in range(len(tau_rat_vals)):
    tau_e = tau_rat_vals[i]*tau_n
    res = integrate(rhs, t_span, initials, t_eval,j,beta,mu_prime,tau_n,tau_e,p,nprime)
    plot_runs(res, tau_n, n_initial, mu_prime, tau_rat_vals[i], t_final, j, beta, t_eval, nprime)