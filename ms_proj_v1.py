import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
from itertools import product
from scipy.optimize import minimize

# ok so once able to replicate their results
# we can try refitting the model (initially performed for SARS) to different diseases (measles, influenza, COVID-19)
# this will be done using the same general model and using MLE to
# estimate new parameters corresponding to different diseases
# fit to part of the data and test predict capability on held out time windows
# validate inferred metrics against published measurements/estimates
"""
Variable | Description
S Population of unvaccinated susceptible individuals
V Population of vaccinated susceptible individuals
E Population of unvaccinated exposed individuals
EV Population of exposed vaccinated individuals
I Population of unvaccinated infectious (symptomatic) individuals
IV Population of infectious vaccinated individuals
Q Population of unvaccinated quarantined individuals
QV Population of quarantined vaccinated individuals
H Population of unvaccinated hospitalized individuals
HV Population of hospitalized vaccinated individuals
R Population of unvaccinated recovered individuals
RV Population of recovered vaccinated individuals


Parameter | Description
Π -> M Recruitment rate
β -> B Effective contact rate
µ -> u Natural death rate
p Fraction of newly-recruited individuals vaccinated
η -> n Modification parameter for reduction in infectiousness of hospitalized individuals
v1 , v2 Modification parameters for reduction in infectiousness of vaccinated infectious and hospitalized individuals
ε -> e Efficacy of vaccine
ζ -> z Vaccination rate of susceptible individuals
ψ -> w Waning rate of vaccine
κ -> k Progression rate from exposed to infectious class
σ -> o0 Quarantine rate for exposed individuals
σ1 -> o1 Quarantine rate for vaccinated exposed individuals
α -> a Hospitalization rate for quarantined individuals
φ -> x Hospitalization rate for infectious individuals
γ1 -> y1 Recovery rate for non-hospitalized infectious individuals
γ2 -> y2 Recovery rate for hospitalized individuals
δ1 -> d1 Disease-induced death rate for non-hospitalized infectious individuals
δ1 -> d2 Disease-induced death rate for hospitalized individuals
θ1,...,θ7 -> m1, . . . , m7 Modification parameters


Acronyms
DFE -> Disease Free Equilibrium
LAS -> Locally-Asymptotically Stable
EEP -> Endemic Equilibrium Point

"""

# This model extends the SEIQHR model defined in [], by introducing 6 new comparments
# for vaccinated invidiuals (V, E_v, Q_v, I_v, H_v, R_v). Its purpose is for
# qualitative analysis and determining the affect of an imperfect vaccine on equilibrium dynamics
# 0 < e < 1, efficacy of vaccine term



def param_dict(use_case):

    # backwards bifurcation #10 ** -7
    if use_case == 'bb': 
        # M, B, p, u, y1, y2, d1, d2, k, a, x, o0, o1, w, n, v1, v2, e, z, m1, m2, m3, m4, m5, m6, m7
        return {'M':136, 'B':1.4, 'p':0.1, 'u':0.001, 'y1':0.01, 'y2':0.1, 'd1':0.001, 
                'd2':0.01, 'k':0.00016, 'a':1, 'x':1, 'o0':1, 'o1':0.09, 'w':0.0001, 
                'n':1, 'v1':0.9, 'v2':1 ,'e':10 ** -7, 'z':0.06, 'm1':0.9, 'm2':1, 
                'm3':0.01, 'm4':1.286, 'm5':0.9, 'm6':1, 'm7':1}
    
    elif use_case == 'oo':  
        # M, B, p, u, y1, y2, d1, d2, k, a, x, o0, o1, w, n, v1, v2, e, z, m1, m2, m3, m4, m5, m6, m7
        return {'M':136, 'B':1.4, 'p':0.1, 'u':0.001, 'y1':0.03870392, 'y2':0.09616842, 'd1':0.001, 
                'd2':0.01, 'k':0.00016, 'a':1, 'x':1, 'o0':1, 'o1':0.09, 'w':0.0001, 
                'n':1, 'v1':0.9, 'v2':1 ,'e':10 ** -7, 'z':0.06, 'm1':0.9, 'm2':1, 
                'm3':0.01, 'm4':1.286, 'm5':0.9, 'm6':1, 'm7':1}

    #infected individuals 
    elif use_case == 'ii':
        # M, B, p, u, y1, y2, d1, d2, k, a, x, o0, o1, w, n, v1, v2, e, z, m1, m2, m3, m4, m5, m6, m7
        return {'M':136, 'B':0.15, 'p':0.25, 'u':0.0000351, 'y1':0.3521, 'y2':0.042553, 'd1':0, 
                'd2':0, 'k':0.156986, 'a':0.156986, 'x':0.20619, 'o0':0.1, 'o1':0.06, 'w':0.0666, 
                'n':0.8, 'v1':0.9, 'v2':0.8 ,'e':0.5, 'z':0.7, 'm1':0.6, 'm2':1.4, 
                'm3':0.7, 'm4':0.6, 'm5':0.5, 'm6':1.4, 'm7':0.7}

def parameter_grid(base_params, param_ranges):
    """
    param_ranges: dict of parameter -> iterable of values to sweep
    base_params: dict of all base parameters
    yields dictionaries of parameters for each combination
    """
    keys = list(param_ranges.keys())
    values = list(param_ranges.values())

    for combo in product(*values):
        p = dict(base_params)
        # Overwrite with parameters of interest
        for k, v in zip(keys, combo):
            p[k] = v
        yield p

# Deterministic ODEs (also may be worth investigating the stochastic case )
def disease_model(tot_pop, t, M, B, p, u, y1, y2, d1, d2, k, a, x, o0, o1, w, n, v1, v2, e, z, m1, m2, m3, m4, m5, m6, m7): 
    S, V, E, EV, I, IV, Q, QV, H, HV, R, RV = tot_pop
    N = S + V + E + EV + I + IV + Q + QV + H + HV + R + RV # initial total human population
    
    # λ -> l defines the rate at which unvaccinated susceptible individuals may acquire infection, 
    # following effective contact with infectious individuals
    l = B * (I + (v1 * IV) + (n * H) + (v2 * HV)) / N

    return [ 
        ((1 - p) * M)  +  (w * V) - (l * S) - ((z + u) * S), # dS/dt
        (p * M)  +  (z * S)  -  ((1 - e) * l * V)  -  ((w + u) * V), # dV/dt
        (l * S)  -  ((k + o0 + u) * E), # dE/dt
        ((1 - e) * l * V)  -  (((m1 * k) + o1 + u) * EV), # dEV/dt
        (k * E)  -  ((y1 + x + u + d1) * I), # dI/dt
        (m1 * k * EV) - (((m2 * y1)  + (m3 * x)  +  u  +  (m4 * d1)) * IV), # dIV/dt
        (o0 * E)  -  ((a  +  u) * Q), # dQ/dt
        (o1 * EV) - (((m5 * a) + u) * QV), # dQV/dt
        (a * Q ) + (x * I) - ((y2 + u + d2) * H), # dH/dt
        (m5 * a * QV) + (m3 * x * IV) - ((m6 * y2 + u + m7 * d2) * HV), # dHV/dt
        (y1 * I) + (y2 * H) - (u * R), #dR/dt
        (m2 * y1 * IV) + (m6 * y2 * HV) - (u * RV) # dRV/dt
    ]

def build_coefficients(B, p, u, y1, y2, k, a, x, o0, o1, w, n, v1, v2, e, z, m1, m2, m3, m4, m5, m6, m7, k1, k2, k3, k4, k5, k6, k7, k8, rvac):
        # a0
    a0 = (
        k2*k4*k6*k8*(1 - e)*(1 - p)*(   
            o0*k3*k7*u + y1*k*k5*k7 + k*k7*k5*u + u*a*o0*k3
        )
        + k2*k4*k6*k8*(1 - e)*(1 - p)*(
            k3*k5*k7*u + y2*x*k5*k + u*x*k5*k + y2*a*o0*k3
        )
        + p*k1*k3*k5*k7*(1 - e)*(
            u*m4*x*k6*m1*k + u*m1*k*k6*k8 + m7*y2*m6*a*m2*o0*k4 +
            m7*y2*m4*x*k6*m1*k
        )
        + p*k1*k3*k5*k7*(1 - e)*(
            m3*y1*m1*k*k6*k8 + u*k4*k6*k8 + u*m2*o0*k4*k8 +
            u*m6*a*m2*o0*k4
        )
    )

    # a1
    a1 = (
        -B*u*(1 - e)*(
            n*k*x*k2*k4*k5*k6*k8*(1 - p)
            + v2*p*n*m5*a*o1*k1*k3*k4*k5*k7
            + n*o0*a*k2*k3*k4*k6*k8*(1 - p)
            + v1*p*m1*k*k1*k3*k5*k6*k7*k8
            + v2*n*p*m1*k*m3*x*k1*k3*k5*k6*k7
            + k*k2*k4*k5*k6*k7*k8*(1 - p)
        )
        + k1*k3*k5*k7*(1 - e)*(
            u*z*m1*k*m3*x*k6*(1 - p)
            + u*z*o1*m5*a*k4*(1 - p)
            + p*u*m1*k*k4*k6*(u + z)
            + p*u*k4*k6*k8*(u + z)
            + u*k2*k4*k6*k8*(1 - p)*(1 + z)
            + u*z*m1*k*k6*k8*(1 - p)
            + z*m1*k*m2*y1*k6*k8*(1 - p)
            + u*z*o1*k4*k8*(1 - p)
            + u*p*o1*k4*k8*(u + z)
            + o1*m5*a*m6*y2*k4
            + u*p*m1*k*m3*x*k6*(u + z)
            + u*p*o1*m5*a*k4*(u + z)
            + z*m1*k*m3*x*m6*y2*k6*(1 - p)
            + p*m1*k*m3*x*m6*y2*k6*(u + z)
            + p*m1*k*m2*y1*k6*k8*(u + z)
            + p*o1*m5*a*m6*y2*k4*(u + z)
        )
        + k2*k4*k6*k8*(
            u*p*k1*k3*k5*k7
            + u*p*w*o0*k3*k5
            + u*p*w*a*o0*k3
            + u*o0*k3*k7*(1 - p)*(u + w)
            + u*p*x*k*x*k5
            + u*k*k5*k7*(u + w)*(1 - p)
            + u*p*w*k*k5*k7
            + u*a*o0*k3*(u + w)*(1 - p)
            + y1*k*k5*k7*(u + w)*(1 - p)
            + p*w*k*y1*k5*k7
            + u*k3*k5*k7*(u + w)*(1 - p)
            + u*x*k*k5*(u + w)*(1 - p)
            + y2*a*o0*k3*(u + w)*(1 - p)
            + p*w*a*o0*y2*k3
            + p*w*k*x*y2*k5
            + u*p*w*k3*k5*k7
            + (1 - p)*(u + w*k*x*y2*k5)
        )
    )

    a2 = u*k1*k2*k3*k4*k5*k6*k7*k8*(u + z + w)*(1 - rvac)
    return a0, a1, a2

def build_infection_matrix(B, n, v1, v2, e, w1, w2):
    F = np.array([
        [0, 0, (B * w1), (v1 * B * w1), 0, 0, (n * B * w1), (v2 * n * B * w1)],
        [0, 0, ((1 - e) * B * w2), ((1 - e) * v1 * B * w2), 0, 0, ((1 - e) * n * B * w2), ((1 - e) * v2 * n * B * w2)],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ])
    return F

def build_transition_matrix(k, a, x, o0, o1, m1, m3, m5, k1, k2, k3, k4, k5, k6, k7, k8):
    G = np.array([
        [k1, 0, 0, 0, 0, 0, 0, 0],
        [0, k2, 0, 0, 0, 0, 0, 0],
        [-k, 0, k3, 0, 0, 0, 0, 0],
        [0, -m1 * k, 0, k4, 0, 0, 0, 0],
        [-o0, 0, 0, 0, k5, 0, 0, 0],
        [0, -o1, 0, 0, 0, k6, 0, 0],
        [0, 0, -x, 0, -a, 0, k7, 0],
        [0, 0, 0, -m3 * x, 0, -m5 * a, 0, k8]
    ])
    return G

def parameter_search(base_params, param_ranges):

    valid_sets = []
    for p_dict in parameter_grid(base_params, param_ranges):
        M, B, p, u, y1, y2, d1, d2, k, a, x, o0, o1, w, n, v1, v2, e, z, m1, m2, m3, m4, m5, m6, m7 = p_dict['M'], p_dict['B'], p_dict['p'], p_dict['u'], p_dict['y1'], p_dict['y2'], p_dict['d1'], p_dict['d2'], p_dict['k'], p_dict['a'], p_dict['x'], p_dict['o0'], p_dict['o1'], p_dict['w'], p_dict['n'], p_dict['v1'], p_dict['v2'], p_dict['e'], p_dict['z'], p_dict['m1'], p_dict['m2'], p_dict['m3'], p_dict['m4'], p_dict['m5'], p_dict['m6'], p_dict['m7']      
        # R vac from spectral radius (largest magnitude eigenvalue)
        w1 = (((1-p) * u) + w) / (u + w + z)
        w2 = ((p * u) + z) / (u + w + z)
        k1 = k + o0 + u
        k2 = (m1 * k) + o1 + u
        k3 = y1 + x + u + d1
        k4 = (m2 * y1) + (m3 * x) + u + (m4 * d1)
        k5 = a + u
        k6 = (m5 * a) + u
        k7 = y2 +  u + d2
        k8 = (m6 * y2) + u + (m7 * d2)

        F = build_infection_matrix(B, n, v1, v2, e, w1, w2)
        G = build_transition_matrix(k, a, x, o0, o1, m1, m3, m5, k1, k2, k3, k4, k5, k6, k7, k8)
        
        
        # control reproduction number from spectral radius
        rvac = max(abs(np.linalg.eigvals(F @ np.linalg.inv(G))))
        #print(f"e={e}; (Control Reproduction Number) R_vac={rvac}")
    
        a0, a1, _ = build_coefficients(B, p, u, y1, y2, k, a, x, o0, o1, w, n, v1, v2, e, z, m1, m2, m3, m4, m5, m6, m7, k1, k2, k3, k4, k5, k6, k7, k8, rvac)

        rcrit_vac = 1 - ((a1 ** 2) / ((4 * a0 * u * k1 * k2 * k3 * k4 * k5 * k6 * k7 * k8) * (u + z + w)))
        
        print(f'e = {e}, rcrit={round(rcrit_vac,2)} < rvac={round(rvac,2)} = {rcrit_vac < rvac}')

        if (rcrit_vac > 0) and (rcrit_vac < 1) and (rvac > rcrit_vac) and (rvac < 1):
            valid_sets.append((p_dict, rcrit_vac, rvac))

    return valid_sets
                

# for figure 2 
def bb_figure(base_params: dict, param_sweep: dict | None = None, search: bool = True):

    # set-up plot
    fig4, ax4 = plt.subplots()
    
    # initialize lists for scatter plot
    Rvac_vals = []
    lambda_vals = []

    if search:
        # running parameter search to identify valid backward bifurcation set
        valid_bb_param = parameter_search(base_params, param_sweep)
    else: 
        valid_bb_param = base_params


    for i in valid_bb_param:
        # using parameters identified from param search
        p_dict,_,rvac = i

        # gross, unpacking the dictionary b/c I don't want to change all my variable names
        M, B, p, u, y1, y2, d1, d2, k, a, x, o0, o1, w, n, v1, v2, e, z, m1, m2, m3, m4, m5, m6, m7 = p_dict['M'], p_dict['B'], p_dict['p'], p_dict['u'], p_dict['y1'], p_dict['y2'], p_dict['d1'], p_dict['d2'], p_dict['k'], p_dict['a'], p_dict['x'], p_dict['o0'], p_dict['o1'], p_dict['w'], p_dict['n'], p_dict['v1'], p_dict['v2'], p_dict['e'], p_dict['z'], p_dict['m1'], p_dict['m2'], p_dict['m3'], p_dict['m4'], p_dict['m5'], p_dict['m6'], p_dict['m7']      
        
        #print(f"e={e}; (Control Reproduction Number) R_vac={rvac}")

        k1 = k + o0 + u
        k2 = (m1 * k) + o1 + u
        k3 = y1 + x + u + d1
        k4 = (m2 * y1) + (m3 * x) + u + (m4 * d1)
        k5 = a + u
        k6 = (m5 * a) + u
        k7 = y2 +  u + d2
        k8 = (m6 * y2) + u + (m7 * d2)
  
        a0, a1, a2 = build_coefficients(B, p, u, y1, y2, k, a, x, o0, o1, w, n, v1, v2, e, z, m1, m2, m3, m4, m5, m6, m7, k1, k2, k3, k4, k5, k6, k7, k8, rvac)
         
        coeffs = [a0, a1, a2]
        roots = np.roots(coeffs)
        print(f"e = {e}; roots = {roots}")

        for lam in roots:
            if np.isreal(lam) and lam.real >= 0:
                Rvac_vals.append(rvac)
                lambda_vals.append(lam.real)


    #print("All roots:", roots)
    #print("Biologically feasible λ** =", lambda_star)
    ax4.scatter(Rvac_vals, lambda_vals)
    ax4.set_xlabel("R_vac")
    ax4.set_ylabel("Force of infection, λ")
    ax4.set_title("Backward Bifurcation Diagram for the Model")
    fig4.tight_layout()

    fig4.savefig("backward_bifurcation.png")
    '''
    # time scale
    t = np.linspace(0, 100, 1000)

    # recreating figure 2
    fig2, ax2 = plt.subplots()

    for i in range(100):
        # initial conditions
        inits = np.random.uniform(100, 1000, size=12)

        # force of infection - same definition as lambda in disease_model()
        # order of initial conditions / returned  compartments -> S, V, E, EV, I, IV, Q, QV, H, HV, R, RV
        f_o_i = B * (inits[4] + (v1 * inits[5]) + (n * inits[8]) + (v2 * inits[9])) / np.sum(inits)

        results = scipy.integrate.odeint(disease_model, inits, t, args=args)

        

        # plotting control reproduction num againt force of infection
        # to observe backwards bifurcation
        ax2.plot(R_vac, f_o_i)

    ax2.set_xlabel("R_vac")
    ax2.set_ylabel("Force of infection, λ")
    ax2.set_title("Backward Bifurcation Diagram for the Model")
    fig2.tight_layout()

    fig2.savefig("backward_bifurcation.png")
    '''

# for figure 3
def ii_figure(p_dict): 
    # Parameter Value Initialization: Infected Individuals

    M, B, p, u, y1, y2, d1, d2, k, a, x, o0, o1, w, n, v1, v2, e, z, m1, m2, m3, m4, m5, m6, m7 = p_dict['M'], p_dict['B'], p_dict['p'], p_dict['u'], p_dict['y1'], p_dict['y2'], p_dict['d1'], p_dict['d2'], p_dict['k'], p_dict['a'], p_dict['x'], p_dict['o0'], p_dict['o1'], p_dict['w'], p_dict['n'], p_dict['v1'], p_dict['v2'], p_dict['e'], p_dict['z'], p_dict['m1'], p_dict['m2'], p_dict['m3'], p_dict['m4'], p_dict['m5'], p_dict['m6'], p_dict['m7']      
    args = (M, B, p, u, y1, y2, d1, d2, k, a, x, o0, o1, w, n, v1, v2, e, z, m1, m2, m3, m4, m5, m6, m7)
 
    # time scale
    t = np.linspace(0, 100, 1000)

    # recreating figure 3
    fig3, ax3 = plt.subplots()

    for i in range(10):
    # initial conditions
        inits = np.random.uniform(100, 1000, size=12)

        # force of infection - same definition as lambda in disease_model()
        f_o_i = B * (inits[4] + (v1 * inits[5]) + (n * inits[8]) + (v2 * inits[9])) / np.sum(inits)
        results = scipy.integrate.odeint(disease_model, inits, t, args=args)

        # order of returned  compartments -> S, V, E, EV, I, IV, Q, QV, H, HV, R, RV
        infected_individuals = results[:,4] + results[:,5]

        # plotting infected individuals dynamics
        ax3.plot(t, infected_individuals)

    ax3.set_xlabel("Time (days)")
    ax3.set_ylabel("Total number of infected individuals")
    ax3.set_title("Steady State Analysis")
    fig3.tight_layout()

    fig3.savefig("infected_individuals.png")


def compute_Rvac(p_dict):

    # gross, unpacking the dictionary b/c I don't want to change all my variable names
    M, B, p, u, y1, y2, d1, d2, k, a, x, o0, o1, w, n, v1, v2, e, z, m1, m2, m3, m4, m5, m6, m7 = p_dict['M'], p_dict['B'], p_dict['p'], p_dict['u'], p_dict['y1'], p_dict['y2'], p_dict['d1'], p_dict['d2'], p_dict['k'], p_dict['a'], p_dict['x'], p_dict['o0'], p_dict['o1'], p_dict['w'], p_dict['n'], p_dict['v1'], p_dict['v2'], p_dict['e'], p_dict['z'], p_dict['m1'], p_dict['m2'], p_dict['m3'], p_dict['m4'], p_dict['m5'], p_dict['m6'], p_dict['m7']      

    w1 = (((1-p) * u) + w) / (u + w + z)
    w2 = ((p * u) + z) / (u + w + z)

    k1 = k + o0 + u
    k2 = (m1 * k) + o1 + u
    k3 = y1 + x + u + d1
    k4 = (m2 * y1) + (m3 * x) + u + (m4 * d1)
    k5 = a + u
    k6 = (m5 * a) + u
    k7 = y2 +  u + d2
    k8 = (m6 * y2) + u + (m7 * d2)

    F = build_infection_matrix(B, n, v1, v2, e, w1, w2)
    G = build_transition_matrix(k, a, x, o0, o1, m1, m3, m5, k1, k2, k3, k4, k5, k6, k7, k8)

    rvac = max(abs(np.linalg.eigvals(F @ np.linalg.inv(G))))

    a0, a1, a2 = build_coefficients(B, p, u, y1, y2, k, a, x, o0, o1, w, n, v1, v2, e, z, m1, m2, m3, m4, m5, m6, m7, k1, k2, k3, k4, k5, k6, k7, k8, rvac)

    rcrit_vac = 1 - ((a1 ** 2) / ((4 * a0 * u * k1 * k2 * k3 * k4 * k5 * k6 * k7 * k8) * (u + z + w)))
    # control reproduction number from spectral radius
    
    return rvac, rcrit_vac, a0, a1, a2

def feasibility_penalty(Rvac, Rcrit, a0, a1, a2):
    # backwards bifurcation critera 
    # logic implemented to penalize optimizer
    # when it strays from necessary conditions
    penalty = 0

    '''
    # push Rcrit > 0
    if Rcrit <= 0:
        penalty += (abs(Rcrit) + 1)**2

    # push Rvac < 1
    if Rvac >= 1:
        penalty += (Rvac - 1 + 1)**2

    # push Rvac > Rcrit
    if Rvac <= Rcrit:
        penalty += (abs(Rcrit - Rvac) + 1)**2
    '''
    # push discriminant > 0
    if (a1 ** 2) - (4 * a0 * a2) <= 0:
        penalty += (a1 + 1.0)**2

    # push a1 < 0 
    if a1 >= 0:
        penalty += (a1 + 1.0)**2

    # push a2 > 0 
    if a2 <= 0:
        penalty += (a1 + 1.0)**2

    return penalty

def loss(theta_vec):
    #  initialize penalty
    total_penalty = 0

    # initialize vec of e values
    e_vec = np.linspace(0, 1, 100, endpoint=False) 

    for e in e_vec:
        pa = dict(param_dict('bb'))
        # unpack theta_vec
        #pa['w'], pa['B'], pa['z'], pa['p'], pa['a'], pa['x'] = theta_vec
        pa['w'], pa['B'], pa['z'], pa['p'], pa['v1'], pa['v2'], pa['m1'], pa['m2'], pa['m3'], pa['m4'], pa['m5'], pa['m6'], pa['m7'] = theta_vec
        pa['e'] = e
        Rvac, Rcrit, a0, a1, a2 = compute_Rvac(pa)

        # have optimizer search for params satisfying backward bifurcation
        # for all e in e_grid
        total_penalty += feasibility_penalty(Rvac, Rcrit, a0, a1, a2)

    return total_penalty

def main(): 

    #x0 = [0.0001, 0.2, 0.06, 0.1]  # initial guesses for (w, B, z, p)
    # loss got stuck (1976.8) expanding to more parameters
    #x0 = [0.0001, 0.2, 0.06, 0.1, 1, 1]  # initial guesses for (w, B, z, p, a, x)
    #bounds = [(0.0001, 0.0666), (0.1, 1.4), (0.06, 0.7), (0.1, .99), (0.156986, 1), (0.20619, 1)]  # sensible bounds

    # loss got stuuuuuck (90)so we're trying just the modification parameters
    #x0 = [0.6, 1.4, 0.7, 0.6, 0.5, 1.4, 0.7]  # initial guesses for (m1, m2, m2, m4, m5, m6, m7)
    #bounds = [(0, 1), (1, 2), (0, 1), (0, 1), (0, 1), (1,2), (0, 1)]  # sensible bounds
    x0 = [0.0001, 2, 0.06, 0.1, 0.9, 1, 0.6, 1.4, 0.7, 0.6, 0.5, 1.4, 0.7]  # initial guesses for (w, B, z, p, v1, v2, m1, m2, m2, m4, m5, m6, m7)

    bounds = [(0.0001, 0.0666), (0.1, np.inf), (0.06, 0.7), (0.1, .99), (0, 1), (0, 1), (0, 1), (1, np.inf), (0, 1), (0, 1), (0, 1), (1,np.inf), (0, 1)]  # sensible bounds
    res = minimize(loss, x0, bounds=bounds, method="L-BFGS-B")
    print(res.x)
    print("Final loss:", res.fun)

    # ok now i fix everything except 'e' -> control parameter, 
    # sweep over 'e' in [0,1] to recreate figure 2

    base = param_dict('bb')

    #base['w'], base['B'], base['z'], base['p'], base['a'], base['x'] = res.x
    base['w'], base['B'], base['z'], base['p'], base['v1'], base['v2'], base['m1'], base['m2'], base['m3'], base['m4'], base['m5'], base['m6'], base['m7'] = res.x
    
    param_to_sweep = {'e':np.linspace(0, 1, 50, endpoint=False)
                      }
    bb_figure(base, param_to_sweep) # backwards bifurcation fig 2
    #ii_figure(param_dict('ii')) # infected individuals fig 3

    
    

if __name__ == "__main__":
    main()

# the parameters needed to numerically demonstrate backwards bifurcation are
# unrealistic, though the authors do assert that since the phenomena only
# exists when the effects of vaccines are taken into consideration, which thus
# shows that adding vaccination to the SEIQHR model alters its qualitative properties