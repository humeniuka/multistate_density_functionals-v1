
Approximate Functionals for Multistate Density Functional Theory
----------------------------------------------------------------
This python package implements approximate multistate matrix functionals for
the electron-electron repulsion and the kinetic energy.
For a single electronic state they reduce to the local density approximation and the
Thomas-Fermi-von Weizsäcker kinetic energy functional.

Requirements
------------

Required python packages:

 * becke-multicenter-integration
 * matplotlib, numpy, pandas, psutil, scipy, tqdm
 * pyscf

A conda environment with the required packaged can be created with

.. code-block:: bash

   $ conda env create -f environment.yml
   $ conda activate msdft

Installation
------------
The package is installed with

.. code-block:: bash

   $ pip install -e .

in the top directory. To verify the proper functioning of the code
a set of tests should be run with

.. code-block:: bash

   $ cd tests
   $ python -m unittest

Getting Started
---------------
An example calculation of LiF is provided in the folder ``examples/09-lithium-fluoride-lda/``.
There are scripts for evaluating the LDA-like multistate density functional along the
dissociation curve of lithium fluoride.

----------
References
----------
.. [1] Yangyi Lu, Jiali Gao, "Multistate Density Functional Theory for Excited States",
    J. Phys. Chem. Lett. 2022, 13, 7762-7769,
    https://doi.org/10.1021/acs.jpclett.2c02088
.. [2] Alexander Humeniuk, "Approximate Functionals for Multistate Density Functional Theory",
    ChemRxiv (2024)
