# Supervised Random Projections with Light
Python implementation of supervised PCA, supervised random projections, and their kernel counterparts.

[Supervised Random Pojections](https://arxiv.org/abs/1811.03166) (SRP) is the work of Amir-Hossein Karimi, Alexander Wong, and Ali Ghodsi. It is a fast approximation of the Supervised PCA algortithm for dimensionality reduction. It also has a nonlinear version, Kernel SRP (KSRP).

This repository provides a unified implementation of SPCA, KSPCA, SRP and KSRP. They are implemented as scikit-learn transformers, and can therefore be used exactly like scikit-learn's PCA and KPCA. Moreover, SRP and KSRP can be performed using a LigthOn Optical Processing Unit (OPU).

- `dimreduc.py` contains the implementations of the algorithms;
- `load_data.py` contains utilities to load the datasets used in the original paper (XOR, Spirals, Sonar and Ionosphere);
- `sonar_viz.py` shows how to use this code for visualizing the Sonar dataset.

The Ionosphere and Sonar dataset come from the UCI repository. They are tiny, so I included them in the `data` folder for convenience.