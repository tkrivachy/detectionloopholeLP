"""
Authors: Tamás Kriváchy, Martin Kerschbaumer
Date: 2025-04-02

Accompanying code for the paper:
Closing the detection loophole in the triangle network with high-dimensional photonic states, Tamás Kriváchy, Martin Kerschbaumer, arXiv:2503.24213 (2025).

preprint URL: https://arxiv.org/abs/2503.24213
preprint DOI: https://doi.org/10.48550/arXiv.2503.24213

Please cite the paper if you use this code.

best_eta_finder.py is written to load a dictionary of sympy probabilities from a pickle file.
Moreover, and index_dict is loaded from a pickle file. This dictionary contains the mapping between the indices of the probabilities and the outcomes.

Example index_dicts:
click/no-click detectors:   {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)}
number resolving detectors: {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (0, 2), 4: (1, 1), 5: (2, 0), 6: (0, 4)}
(here all outcomes with 3 or 4 photons arriving at a party are grouped together as (0,4))

ostarLP.py contains only the LP optimization function, which just needs the float probabilities, all indices, and the ostar index.

Tested with Python 3.11.4 using the following packages:
numpy version: 2.0.2
sympy version: 1.13.3
matplotlib version: 3.10.0
pickle version: 4.0
picos version: 2.6.0
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import pickle
import time
from ostarLP import run_ostar_optimization
import sys

np.set_printoptions(precision=5,suppress=True,linewidth=200)
sys.setrecursionlimit(9999)

def load_output_probabilities_noisy(filename_probs, filename_index_dict):
    with open(filename_probs, 'rb') as file:
        output_probs = pickle.load(file)
    with open(filename_index_dict, 'rb') as file:
        index_dict = pickle.load(file)
    try:
        for key, value in output_probs.items():
            if value != 0:
                # Define sympy vars that appear in the dictionary
                [sp.var(str(i), **i.assumptions0) for i in value.atoms(sp.Symbol)]
    except:
        print("Loading non-sympy dictionary.")
            
    return output_probs, index_dict

####### CONFIG #######
###### Change comments to switch between different scans ######

# 2 photons, click no click detectors eta and t scan w/ 1 photons lost
# filename_probs = './saved_probs/saved_probs_N_2_max_1_photon_lost_coarse_grained_click-noclick.pkl'
# filename_index_dict = './saved_probs/saved_index_dicts_N_2_max_1_photon_lost_coarse_grained_click-noclick.pkl'
# ostar_index = 0 # we want to have (0,0) as ostar (see index_dict for which index is which outcome)

# 2 photons, number resolving detectors eta and t scan w/ 1 photons lost
filename_probs = './saved_probs/saved_probs_N_2_number_resolvng_max_1_photon_lost_coarse_grained_3_and_higher.pkl'
filename_index_dict = './saved_probs/saved_index_dicts_N_2_number_resolvng_max_1_photon_lost_coarse_grained_3_and_higher.pkl'
ostar_index = 0 # we want to have (0,0) as ostar (see index_dict for which index is which outcome)

# 2 photons, number resolving detectors eta and t scan w/ all photons lost
# filename_probs = './saved_probs/saved_probs_N_2_number_resolvng_all_photons_lost_coarse_grained_3_and_higher.pkl'
# filename_index_dict = './saved_probs/saved_index_dicts_N_2_number_resolvng_all_photons_lost_coarse_grained_3_and_higher.pkl'
# ostar_index = 6 # we want to have (0,4) as ostar (see index_dict for which index is which outcome)

output_probs, index_dict = load_output_probabilities_noisy(filename_probs=filename_probs, filename_index_dict=filename_index_dict)

# Substitutions are constant
substitutions = {phi: np.pi/2,
                tilt: 0}

# Scan and bisection params are changed from round to round. 
# ALSO change them in params_to_distribution function!
param_scan_name = 't'
param_bisection_name = 'eta'
param_scan_range = np.linspace(0.75,1,11) # nonlocal between ~0.76 and 1
param_bisection_local_start = 0.5 # eta value for which it is for sure local
param_bisection_nonlocal_start = 1 # eta value for which it is for sure nonlocal
desired_precision = 0.0001

def params_to_distribution(param_scan, param_bisection, output_probs):
    scan_and_bisection_substitutions = {
        t: param_scan,
        eta: param_bisection
    }
    try:
        output_probs_numeric = {key: float(value.subs(scan_and_bisection_substitutions).evalf()) for key, value in output_probs.items()}
    except:
        output_probs_numeric = {}
        try:
            for key, value in output_probs.items():
                if type(value) != int and type(value) != float:
                    output_probs_numeric[key] = float(np.abs(value.subs(scan_and_bisection_substitutions).evalf()))
                else:
                    output_probs_numeric[key] = float(np.abs(value))
        except:
            print("Error in substituting fixed params for numerical values.")
    return output_probs_numeric

### END OF CONFIG ###

best_nonlocal_bisection_values = np.ones_like(param_scan_range) * param_bisection_nonlocal_start

# Change all keys to integers
output_probs = {(int(key[0]), int(key[1]), int(key[2])): value for key, value in output_probs.items()}

p_len = len(output_probs)
outcomes_per_party = round(p_len**(1/3))
all_outcome_indices = list(index_dict.keys())
print("Number of events:",p_len)
print("Outcomes per party:",outcomes_per_party)
print(index_dict)


# Substitute fixed values (tilt, phi) for numerical values
try:
    output_probs = {key: value.subs(substitutions) for key, value in output_probs.items()}
except:
    try:
        for key, value in output_probs.items():
            if type(value) != int and type(value) != float:
                output_probs[key] = value.subs(substitutions)
            else:
                output_probs[key] = value
    except:
        print("Error in substituting fixed params for numerical values.")


for i in range(param_scan_range.shape[0]):
    param_scan = param_scan_range[i]
    print()
    print("Starting bisection for scan param ",param_scan)
    current_bisection_local = param_bisection_local_start
    current_bisection_nonlocal = param_bisection_nonlocal_start
    print("1st local = {} and 1st nonlocal = {}:".format(current_bisection_local, current_bisection_nonlocal))
    while np.abs(current_bisection_local-current_bisection_nonlocal)>desired_precision:
        start_time = time.time()
        
        param_bisection = (current_bisection_local+current_bisection_nonlocal)/2
        output_probs_numeric = params_to_distribution(param_scan,param_bisection, output_probs)
        
        this_sol = run_ostar_optimization(output_probs_numeric, all_outcome_indices, ostar_index)
        if "infeasible" in this_sol.lower():
            current_bisection_nonlocal = param_bisection
        else:
            current_bisection_local = param_bisection
        print("New local = {} and new nonlocal = {} completed in {} s.".format(current_bisection_local, current_bisection_nonlocal, time.time() - start_time))
    
    best_nonlocal_bisection_values[i] = current_bisection_nonlocal

    # saveto = f"./bisection_scan_data/best_nonlocal_bisection_values_{param_scan_name}_{param_bisection_name}.csv"
    # np.savetxt(saveto, best_nonlocal_bisection_values)
    # saveto = f"./bisection_scan_data/param_scan_range_{param_scan_name}_{param_bisection_name}.csv"
    # np.savetxt(saveto, param_scan_range)

    # plt.clf()
    # plt.plot(param_scan_range, best_nonlocal_bisection_values)
    # plt.xlabel(param_scan_name)
    # plt.ylabel(param_bisection_name)
    # plt.show()
    # save in eta_Q_LP_figs
    # plt.savefig(f'./bisection_scan_data/scan_{param_scan_name}_{param_bisection_name}.png')
plt.clf()
plt.plot(param_scan_range, best_nonlocal_bisection_values)
plt.xlabel(param_scan_name)
plt.ylabel(param_bisection_name)
plt.show()
# save in eta_Q_LP_figs
# plt.savefig(f'./bisection_scan_data/scan_{param_scan_name}_{param_bisection_name}.png')

print("Max value {} at scan param {}:".format(np.max(best_nonlocal_bisection_values), param_scan_range[np.argmax(best_nonlocal_bisection_values)]))
print("Min value {} at scan param {}:".format(np.min(best_nonlocal_bisection_values), param_scan_range[np.argmin(best_nonlocal_bisection_values)]))
