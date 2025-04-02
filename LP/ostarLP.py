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
import picos as pic

np.set_printoptions(precision=5, suppress=True, linewidth=200)

def run_ostar_optimization(numeric_p, all_outcome_indices, ostar_index, verbose=False):
    """ ostar is the protected outcome (p(o*,o*)=0)
    arguments:
    numeric_p: dict of probabilities (float values)
    all_outcome_indices: list of all outcome indices
    ostar_index: index of the protected outcome
    verbose: bool, if True, prints additional information
    
    returns:
    "feasible" if the problem is feasible, "infeasible" otherwise
    """
    ostar_indices = [ostar_index]
    chi = [eh for eh in all_outcome_indices if eh not in ostar_indices]
    if verbose:
        print(ostar_indices)
        print(chi)
    
    i_max = j_max = k_max = len(chi)
    s_max = 2

    numeric_sum = sum(numeric_p.values())
    assert np.abs(numeric_sum - 1) < 1e-10, "Numerical sum is not 1! It is: " + str(numeric_sum)

    def pt(i, j, k):
        """ Sums over all possible values of i, j, k. """
        if not isinstance(i, list): i = [i]
        if not isinstance(j, list): j = [j]
        if not isinstance(k, list): k = [k]
        return np.sum([numeric_p[(i_, j_, k_)] for i_ in i for j_ in j for k_ in k])

    Q = pic.RealVariable("Q", i_max * j_max * k_max * s_max)
    q = {(i, j, k, s): Q[i * j_max * k_max * s_max + j * k_max * s_max + k * s_max + s]
         for i in range(i_max) for j in range(j_max) for k in range(k_max) for s in range(s_max)}

    R = pic.RealVariable("R", i_max * j_max * k_max * 3)
    r = {(i, j, k, s): R[i * j_max * k_max * 3 + j * k_max * 3 + k * 3 + s]
         for i in range(i_max) for j in range(j_max) for k in range(k_max) for s in range(3)}

    postar = pt(all_outcome_indices, all_outcome_indices, ostar_indices)

    if 1-4*postar==0:
        lb = 1/2
        ub = 1/2
    else:
        lb = 1/2 - 1/2*np.sqrt(np.abs(1-4*postar)) #lower bound for LHV's
        ub = 1/2 + 1/2*np.sqrt(np.abs(1-4*postar)) #upper bound for LHV's
    lbSUS = 1/4
    ubSUS = (lb**3 + ub**3)

    if verbose:
        print("postar: ", postar)
        print("lb: ", lb)
        print("ub: ", ub)
        print("lbSUS: ", lbSUS)
        print("ubSUS: ", ubSUS)
        print("ub**2 - postar: ", ub**2 - postar)
        print("lb**2 - postar: ", lb**2 - postar)

    constraints = []
    
    # Constraint 0 and 1:
    constraints.append(sum(q[i, j, k, s] for i in range(i_max) for j in range(j_max) for k in range(k_max) for s in range(s_max)) == 1)
    for i in range(i_max):
        for j in range(j_max):
            for k in range(k_max):
                # Constraint 0:
                for s in range(s_max):
                    constraints.append(q[i, j, k, s] >= 0)
                for party in range(3):
                    constraints.append(r[i, j, k, party] >= 0)

                # Constraint 1:
                constraints.append(sum(q[i,j,k,s] for s in range(s_max)) >= 1/ubSUS * (pt(chi[i], chi[j], chi[k]) - sum(r[i, j, k, party] for party in range(3))))
                constraints.append(sum(q[i,j,k,s] for s in range(s_max)) <= 1/lbSUS * (pt(chi[i], chi[j], chi[k]) - sum(r[i, j, k, party] for party in range(3))))
    

    # Constraint 2:
    for party in range(3):
        constraints.append(sum(r[i, j, k, party] for i in range(i_max) for j in range(j_max) for k in range(k_max)) <= ub**2 - postar)
        constraints.append(sum(r[i, j, k, party] for i in range(i_max) for j in range(j_max) for k in range(k_max)) >= lb**2 - postar)
    
    # Constraint 3:
    for i in range(i_max):
        q_i_0 = sum(q[i, j, k, 0] for j in range(j_max) for k in range(k_max))
        q_i_1 = sum(q[i, j, k, 1] for j in range(j_max) for k in range(k_max))
        r_i_B = sum(r[i, j, k, 1] for j in range(j_max) for k in range(k_max))
        r_i_C = sum(r[i, j, k, 2] for j in range(j_max) for k in range(k_max))


        constraints.append(q_i_0 <= ub/lb * q_i_1 + 
                            1/lbSUS * (
                                ub/lb * pt(chi[i],ostar_indices,chi) - pt(chi[i], chi, ostar_indices) +
                                ub/lb * r_i_B - r_i_C
                            ))
        constraints.append(q_i_0 >= lb/ub * q_i_1 +
                            1/ubSUS * (
                                lb/ub * pt(chi[i],ostar_indices,chi) - pt(chi[i], chi, ostar_indices) +
                                lb/ub * r_i_B - r_i_C
                            ))
    
    for j in range(j_max):
        q_j_0 = sum(q[i, j, k, 0] for i in range(i_max) for k in range(k_max))
        q_j_1 = sum(q[i, j, k, 1] for i in range(i_max) for k in range(k_max))
        r_j_C = sum(r[i, j, k, 2] for i in range(i_max) for k in range(k_max))
        r_j_A = sum(r[i, j, k, 0] for i in range(i_max) for k in range(k_max))


        constraints.append(q_j_0 <= ub/lb * q_j_1 + 
                            1/lbSUS * (
                                ub/lb * pt(chi,chi[j],ostar_indices) - pt(ostar_indices, chi[j], chi) +
                                ub/lb * r_j_C - r_j_A
                            ))
        constraints.append(q_j_0 >= lb/ub * q_j_1 +
                            1/ubSUS * (
                                lb/ub * pt(chi,chi[j],ostar_indices) - pt(ostar_indices, chi[j], chi) +
                                lb/ub * r_j_C - r_j_A
                            ))
        
    for k in range(k_max):
        q_k_0 = sum(q[i, j, k, 0] for i in range(i_max) for j in range(j_max))
        q_k_1 = sum(q[i, j, k, 1] for i in range(i_max) for j in range(j_max))
        r_k_A = sum(r[i, j, k, 0] for i in range(i_max) for j in range(j_max))
        r_k_B = sum(r[i, j, k, 1] for i in range(i_max) for j in range(j_max))
        
        constraints.append(q_k_0 <= ub/lb * q_k_1 + 
                            1/lbSUS * (
                                ub/lb * pt(ostar_indices,chi,chi[k]) - pt(chi,ostar_indices,chi[k]) +
                                ub/lb * r_k_A - r_k_B
                            ))
        constraints.append(q_k_0 >= lb/ub * q_k_1 +
                            1/ubSUS * (
                                lb/ub * pt(ostar_indices,chi,chi[k]) - pt(chi,ostar_indices,chi[k]) +
                                lb/ub * r_k_A - r_k_B
                            ))
        
    problem = pic.Problem()
    for c in constraints:
        problem.add_constraint(c)
    problem.set_objective("find")

    try:
        solution = problem.solve(verbosity=False, primals=True)
    except:
        solution = "infeasible"

    # Some examples of how to extract the values
    # solution.apply()
    # print("Printing values")
    # print(R[0].value)
    # print(R.np)
    # for qitem in Q:
    #     print(qitem.value)
    # print(q)
    # print(solution)

    # Double check that hte constraints are satisfied
    # print(q[0,0,0,0].value + q[0,0,0,1].value)
    # print(4*(pt(chi[0], chi[0], chi[0]) - R[0].value - R[1].value - R[2].value))

    return "feasible" if "infeasible" not in str(solution).lower() else "infeasible"