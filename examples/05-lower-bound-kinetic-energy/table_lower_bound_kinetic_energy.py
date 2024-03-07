#!/usr/bin/env python
# coding: utf-8
"""
compares the lower bounds for the subspace kinetic energy with the exact one
and save the table to a csv file.
"""
import numpy
import pandas
import pyscf.gto

from msdft.LowerBoundKinetic import LowerBoundKineticSubspaceInvariant
from msdft.LowerBoundKinetic import LowerBoundKineticSumOverStates
from msdft.MultistateMatrixDensity import MultistateMatrixDensityFCI

# test molecules
molecules = {
    # 1-electron systems
    'H': pyscf.gto.M(
        atom = 'H 0 0 0',
        basis = '6-31g',
        # doublet
        spin = 1),
    'H (large basis set)': pyscf.gto.M(
        atom = 'H 0 0 0',
        basis = 'aug-cc-pvtz',
        # doublet
        spin = 1),
    'H$_2^+$': pyscf.gto.M(
        atom = 'H 0 0 0; H 0 0 0.74',
        basis = '6-31g',
        charge = 1,
        spin = 1),
    # 2-electron systems, paired spins
    'H$_2$': pyscf.gto.M(
        atom = 'H 0 0 0; H 0 0 0.74',
        basis = '6-31g',
        charge = 0,
        spin = 0),
    # 3-electron systems, one unpaired spin
    'Li': pyscf.gto.M(
        atom = 'Li 0 0 0',
        basis = '6-31g',
        # doublet
        spin = 1),
    # 4-electron system, closed shell
    'LiH': pyscf.gto.M(
        atom = 'Li 0 0 0; H 0 0 1.60',
        basis = '6-31g',
        # singlet
        spin = 0),
    # many electrons
    'water': pyscf.gto.M(
        atom = 'O  0 0 0; H 0.75 0.00 0.50; H 0.75 0.00 -0.50',
        basis = 'sto-3g',
        # singlet
        spin = 0),
    # triplet oxygen molecule
    'O$_2$': pyscf.gto.M(
        atom = 'O  0 0 0; O 0 0 1.207',
        basis = 'sto-3g',
        # triplet
        spin = 2),
    'NO': pyscf.gto.M(
        atom = 'N  0 0 0; O 0 0 1.1508',
        basis = 'sto-3g',
        # double
        spin = 1),
    # noble gas atom
    'He': pyscf.gto.M(
        atom = 'He  0 0 0',
        basis = '6-31g',
        # singlet
        spin = 0),
    'Ne': pyscf.gto.M(
        atom = 'Ne  0 0 0',
        basis = '6-31g',
        # singlet
        spin = 0)
    }

dataframe = pandas.DataFrame(columns=['molecule', 'basis', 'bound 1 (sum over states)', 'bound 2 (subspace invariant)', 'exact'])
for name, mol in molecules.items():
    # Compute the matrix density for the lowest 4 states (if available)
    msmd = MultistateMatrixDensityFCI.create_matrix_density(
        mol, nstate=4, raise_error=False)

    # Lower bounds
    # 1) ∑ᵢ 1/8 ∫ |∇Dᵢᵢ|²/Dᵢᵢ(r)
    lower_bound_1 = LowerBoundKineticSumOverStates(mol)(msmd)
    # 2) 1/8 ∫ |∇ρᵥ(r)|²/ρᵥ(r)
    lower_bound_2 = LowerBoundKineticSubspaceInvariant(mol)(msmd)

    # The exact kinetic energy matrix Tᵢⱼ
    T_exact = msmd.exact_kinetic_energy()
    # number of electronic states
    nstate = T_exact.shape[0]
    # exact subspace kinetic energy, 1/N tr(T)
    subspace_kinetic_energy = 1.0/nstate * numpy.trace(T_exact)

    dataframe.loc[len(dataframe)] = [name, mol.basis, lower_bound_1, lower_bound_2, subspace_kinetic_energy]

dataframe.to_csv('kinetic_energy_bounds.csv')
print(dataframe)
