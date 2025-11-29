import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt

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

# Deterministic ODEs
# also may be worth investigating the stochastic case 
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


def main(): 
    #''' 
    # Parameter Value Initialization: SARS Model 
    M = 136
    B = 0.15 #[0.1, 0.2] # variable
    p = 0.25 #[0.25, 0.4] # variable
    u = 0.0000351
    y1 = 0.03521
    y2 = 0.042553
    d1 = 0 #0.04227
    d2 = 0 #0.027855
    k =  0.156986
    a = 0.156986
    x = 0.20619
    o0 = 0.1
    o1 = 0.06
    w = 0.0666
    n = 0.8
    v1 = 0.9
    v2 = 0.8
    e  = 0.5 #[0, 1] # variable
    z  = 0.7
    m1 = 0.6
    m2 = 1.4
    m3 = 0.7
    m4 = 0.6
    m5 = 0.5
    m6 = 1.4
    m7 = 0.7
    #'''
    '''
        # Parameter Value Initialization: SARS Model 
    M = 136
    B = 1.4 #[0.1, 0.2] # variable
    p = 0.25 #[0.25, 0.4] # variable
    u = 0.001
    y1 = 0.03521
    y2 = 0.042553
    d1 = 0 #0.04227
    d2 = 0 #0.027855
    k =  0.00016
    a = 0.156986
    x = 0.20619
    o0 = 0.1
    o1 = 0.09
    w = 0.0001
    n = 0.8
    v1 = 0.9
    v2 = 0.8
    e  = 0.5 #[0, 1] # variable
    z  = 0.06
    m1 = 0.7
    m2 = 1
    m3 = 0.7
    m4 = 0.9
    m5 = 0.9
    m6 = 1.4
    m7 = 0.7
    '''
    # parameters
    args = (M, B, p, u, y1, y2, d1, d2, k, a, x, o0, o1, w, n, v1, v2, e, z, m1, m2, m3, m4, m5, m6, m7)

    # time scale
    t = np.linspace(0, 100, 1000)

    for i in range(10):
        # initial conditions
        inits = np.random.uniform(100, 1000, size=12)

        results = scipy.integrate.odeint(disease_model, inits, t, args=args)

        # order of returned  compartments -> S, V, E, EV, I, IV, Q, QV, H, HV, R, RV
        infected_individuals = results[:,0]

        plt.plot(t, infected_individuals)

    plt.xlabel("Time (days)")
    plt.ylabel("Total number of infected individuals")
    plt.title("ODE Solution")
    plt.tight_layout()
    # if library just show
    # plt.show()
    # save out
    plt.savefig("infected_individuals_fig1.png", dpi=300)

    

if __name__ == "__main__":
    main()
# the parameters needed to numerically demonstrate backwards bifurcation are
# unrealistic, though the authors do assert that since the phenomena only
# exists when the effects of vaccines are taken into consideration, which thus
# shows that adding vaccination to the SEIQHR model alters its qualitative properties