# Closing the detection loophole in the triangle network with high-dimensional photonic states
Authors: Tamás Kriváchy, Martin Kerschbaumer
Date: 2025-04-02

Accompanying code for the paper:
Closing the detection loophole in the triangle network with high-dimensional photonic states, Tamás Kriváchy, Martin Kerschbaumer, arXiv:2503.24213 (2025).

preprint URL: https://arxiv.org/abs/2503.24213

preprint DOI: https://doi.org/10.48550/arXiv.2503.24213

Please cite the paper if you use this code.

## Description

best_eta_finder.py is written to load a dictionary of sympy probabilities from a pickle file. Two examples provided (click/no-click and ph. number resolving for N=2)
Moreover, and index_dict is loaded from a pickle file. This dictionary contains the mapping between the indices of the probabilities and the outcomes.

Example index_dicts:
click/no-click detectors:   {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)}
number resolving detectors: {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (0, 2), 4: (1, 1), 5: (2, 0), 6: (0, 4)}
(here all outcomes with 3 or 4 photons arriving at a party are grouped together as (0,4))

ostarLP.py contains only the LP optimization function, which just needs the float probabilities, all indices, and the ostar index.

## Version info

Tested with Python 3.11.4 using the following packages:
- numpy version: 2.0.2
- sympy version: 1.13.3
- matplotlib version: 3.10.0
- pickle version: 4.0
- picos version: 2.6.0
