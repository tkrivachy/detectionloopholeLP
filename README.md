# Closing the detection loophole in the triangle network with high-dimensional photonic states
Authors: Tamás Kriváchy, Martin Kerschbaumer
Date: 2025-04-02

Accompanying code for the paper:
Closing the detection loophole in the triangle network with high-dimensional photonic states, Tamás Kriváchy, Martin Kerschbaumer, arXiv:2503.24213 (2025).

preprint DOI: https://doi.org/10.48550/arXiv.2503.24213

Please cite the paper if you use the LP code.

## Description

### LP directory
LP/best_eta_finder.py is written to load a dictionary of sympy probabilities from a pickle file. Two examples provided (click/no-click and ph. number resolving for N=2)
Moreover, and index_dict is loaded from a pickle file. This dictionary contains the mapping between the indices of the probabilities and the outcomes.
The script finds the smallest eta value for which there is infeasibility using the bisection method, for multiple values of t.

Example index_dicts:
- click/no-click detectors:   {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)}
- number resolving detectors: {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (0, 2), 4: (1, 1), 5: (2, 0), 6: (0, 4)}; 
(here all outcomes with 3 or 4 photons arriving at a party are grouped together as (0,4))

LP/ostarLP.py contains only the LP optimization function, which just needs the float probabilities, all indices, and the ostar index. Return "feasible" or "infeasible".

### LHVNet directory

If you use this, please cite: A neural network oracle for quantum nonlocality problems in networks, Tamás Kriváchy, Yu Cai, Daniel Cavalcanti, Arash Tavakoli, Nicolas Gisin and Nicolas Brunner, npj Quantum Information 6, 70 (2020)

DOI: https://doi.org/10.1038/s41534-020-00305-x

main github directory for LHVNet codes: https://github.com/tkrivachy/neural-network-for-nonlocality-in-networks

We provide sample code for scanning eta values for click/no-click detectors. Note that in order to reproduce the results in the paper, we recommend to rerun the code multiple times (10-30) and keep the lowest Euclidean distances.

Usage: run train_multiple_sweeps.py. Results are saved in the 0_saved_results directory.

## Version info

LP code tested with Python 3.11.4 using the following packages:
- numpy version: 2.0.2
- sympy version: 1.13.3
- matplotlib version: 3.10.0
- pickle version: 4.0
- picos version: 2.6.0

LHV-Net code additionally used Tensorflow version: 2.18.0.
