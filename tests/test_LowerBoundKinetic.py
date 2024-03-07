#!/usr/bin/env python
# coding: utf-8
from abc import ABC, abstractmethod
import numpy
import pyscf.gto

from tqdm import tqdm
import unittest

from msdft.LowerBoundKinetic import LowerBoundKinetic
from msdft.LowerBoundKinetic import LowerBoundKineticSubspaceInvariant
from msdft.LowerBoundKinetic import LowerBoundKineticSumOverStates
from msdft.MultistateMatrixDensity import MultistateMatrixDensity
from msdft.MultistateMatrixDensity import MultistateMatrixDensityFCI


class VonWeizsaeckerFunctionalSpinSummed(object):
    """
    The von Weizsäcker density functional of the kinetic energy:

                        (∇ρ)²
           T[ρ] = ∫ 1/8 ----
                          ρ

    Note that the kinetic energy is calculated for the sum of the
    spin densities, ρ = ρ(↑↑) + ρ(↓↓).
    """
    def __init__(self, mol, level=8):
        # generate a multicenter integration grid
        self.grids = pyscf.dft.gen_grid.Grids(mol)
        self.grids.level = level
        self.grids.build()

    def __call__(
            self,
            msmd : MultistateMatrixDensity):
        """
        Compute the von Weizsaecker kinetic energy for the density of a single
        electronic state.

        :param msmd: A multistate density matrix with only a single electronic state
        :type msmd: :class:`~.MultistateMatrixDensity`

        :return kinetic_energy: A 1x1 matrix with the scalar kinetic energy.
        :rtype kinetic_energy: numpy.ndarray of shape (1,1)
        """
        # number of electronic states
        nstate = msmd.number_of_states
        assert nstate == 1, \
           "The von Weizsaecker functional is only defined for a single electronic state."
        # up or down spin
        nspin = 2

        # Evaluate D(r) and ∇D(r) on the integration grid.
        D, grad_D, _ = msmd.evaluate(self.grids.coords)
        # sum over spins
        D = numpy.einsum('sijr->ijr', D)
        grad_D = numpy.einsum('sijdr->ijdr', grad_D)

        # von Weizsäcker kinetic energy density
        KED = 1.0/8.0 * (
            numpy.einsum('ikdr,kjdr->ijr', grad_D, grad_D) / D
        )

        # The matrix of the kinetic energy operator in the subspace is obtained
        # by integration T_{i,j}(r) over space
        #
        #  ∫ -1/2 ϕᵢ*(r) ∇²ϕⱼ(r) = ∫ 1/2 ∇ϕᵢ*(r) ∇ϕⱼ(r)
        #
        kinetic_matrix = numpy.einsum('r,ijr->ij', self.grids.weights, KED)

        return kinetic_matrix


class LowerBoundKineticTestCase(ABC, unittest.TestCase):
    """
    Abstract base class for testing the kinetic energy bounds.
    It contains functions needed by all tests.
    """
    @property
    @abstractmethod
    def lower_bound_kinetic_class(self):
        """
        The subclass of :class:`~.LowerBoundKinetic` for which
        the respective unit test is written.
        """
        pass

    def create_test_molecules_1electron(self):
        """ dictionary with 1-electron molecules to run the tests on """
        molecules = {
            # 1-electron systems
            'hydrogen atom': pyscf.gto.M(
                atom = 'H 0 0 0',
                basis = '6-31g',
                # doublet
                spin = 1),
            'hydrogen atom (large basis set)': pyscf.gto.M(
                atom = 'H 0 0 0',
                basis = 'aug-cc-pvtz',
                # doublet
                spin = 1),
            'hydrogen molecular ion': pyscf.gto.M(
                atom = 'H 0 0 0; H 0 0 0.74',
                basis = '6-31g',
                charge = 1,
                # doublet
                spin = 1),
        }
        return molecules

    def create_test_molecules(self):
        """ dictionary with molecules to run the tests on """
        molecules = {
            # 2-electron systems, paired spins
            'hydrogen molecule': pyscf.gto.M(
                atom = 'H 0 0 0; H 0 0 0.74',
                basis = '6-31g',
                charge = 0,
                spin = 0),
            # 4-electron system, closed shell
            'lithium hydride': pyscf.gto.M(
                atom = 'Li 0 0 0; H 0 0 1.60',
                basis = '6-31g',
                # singlet
                spin = 0),
            # open-shell system, unpaired electron
            'lithium atom': pyscf.gto.M(
                atom = 'Li 0 0 0',
                basis = '6-31g',
                # doublet
                spin = 1),
        }
        return molecules

    def create_matrix_density(self, mol, nstate=4):
        """
        Compute multistate matrix density for the lowest few excited states
        of a small molecule using full configuration interaction.

        :param mol: A test molecule
        :type mol: gto.Mole

        :param nstate: number of excited states to calculate
        :type nstate: positive int

        :return: multistate matrix density
        :rtype: MultistateMatrixDensity
        """
        # call static method
        return MultistateMatrixDensityFCI.create_matrix_density(
            mol, nstate=nstate, spin_symmetry=False, raise_error=False)

    def check_is_kinetic_energy_bound(self, mol, nstate=1):
        """
        Check that the lower bound satisfies the inequality

            1/N ∑ᵢ Tᵢᵢ[D] ≥ lower_bound[D]

        :param mol: A test molecule with only one electron.
        :type mol: gto.Mole

        :param nstate: Number of electronic states in the subspace.
           The full CI problem is solved for the lowest nstate states.
        :type nstate: int > 0
        """
        # Check that the derived unit test is implemented correctly.
        assert issubclass(self.lower_bound_kinetic_class, LowerBoundKinetic)

        # functional for lower bound
        lower_bound_functional = self.lower_bound_kinetic_class(mol, level=5)

        # compute D(r) from full CI
        msmd = self.create_matrix_density(mol, nstate=nstate)

        # Evaluate lower_bound[D(r)]
        lower_bound = lower_bound_functional(msmd)

        # The exact kinetic energy matrix is calculated by contracting the (transition)
        # density matrices in the AO basis with the kinetic energy matrix.
        T_exact = msmd.exact_1e_operator(intor='int1e_kin')
        # Number of states in the subspace.
        nstate = T_exact.shape[0]
        # The average kinetic energy in the subspace, 1/N tr(T)
        subspace_kinetic_energy = numpy.trace(T_exact) / nstate

        self.assertLessEqual(lower_bound, subspace_kinetic_energy)

    def check_chunk_size(self, mol):
        """
        Compute lower bound with different chunk sizes.
        """
        # Check that the derived unit test is implemented correctly.
        assert issubclass(self.lower_bound_kinetic_class, LowerBoundKinetic)
        # functional for kinetic operator, T[D(r)]
        lower_bound_functional = self.lower_bound_kinetic_class(mol, level=1)

        # compute D(r) from full CI
        msmd = self.create_matrix_density(mol, nstate=3)
        # lower bound with default chunk size
        lower_bound_ref = lower_bound_functional(msmd)
        # Increase the number of chunks by reducing the available memory
        # per chunk to 2**22 (~4 Mb) or 2**23 (~ 8Mb) bytes.
        for memory in [2**22, 2**23]:
            lower_bound = lower_bound_functional(msmd, available_memory=memory)

            self.assertAlmostEqual(lower_bound_ref, lower_bound)

    def check_single_state_von_Weizsaecker(self, mol):
        """
        For a single electronic state the lower bound should be equal to the
        von Weizsäcker functional.
        """
        # Check that the derived unit test is implemented correctly.
        assert issubclass(self.lower_bound_kinetic_class, LowerBoundKinetic)

        # functional for lower bound
        lower_bound_functional = self.lower_bound_kinetic_class(mol, level=5)

        # compute D(r) from full CI
        msmd = self.create_matrix_density(mol, nstate=1)

        # Evaluate lower_bound[D(r)]
        lower_bound = lower_bound_functional(msmd)

        # vW functional
        kinetic_functional_single = VonWeizsaeckerFunctionalSpinSummed(mol)
        kinetic_matrix_single = kinetic_functional_single(msmd)

        # For a single state the equality holds.
        self.assertAlmostEqual(kinetic_matrix_single[0,0], lower_bound)


class TestLowerBoundKineticSumOverStates(LowerBoundKineticTestCase):
    @property
    def lower_bound_kinetic_class(self):
        """ The functional to be tested. """
        return LowerBoundKineticSumOverStates

    def test_is_lower_bound(self):
        """ Check that the exact kinetic energy is larger than the lower bounds. """
        for name, mol in tqdm(
                self.create_test_molecules().items()):
            for nstate in tqdm([1,2,3,4]):
                with self.subTest(molecule=name, nstate=nstate):
                    self.check_is_kinetic_energy_bound(mol, nstate=nstate)

    def test_single_state_von_Weizsaecker(self):
        """
        Check that for a single electronic state the lower bound is equal to
        the von-Weizsäcker kinetic energy.
        """
        for name, mol in tqdm(
                self.create_test_molecules().items()):
            with self.subTest(molecule=name):
                self.check_single_state_von_Weizsaecker(mol)


class TestLowerBoundKineticSubspaceInvariant(LowerBoundKineticTestCase):
    @property
    def lower_bound_kinetic_class(self):
        """ The functional to be tested. """
        return LowerBoundKineticSubspaceInvariant

    def test_is_lower_bound(self):
        """ Check that the exact kinetic energy is larger than the lower bounds. """
        for name, mol in tqdm(
                self.create_test_molecules().items()):
            for nstate in tqdm([1,2,3,4]):
                with self.subTest(molecule=name, nstate=nstate):
                    self.check_is_kinetic_energy_bound(mol, nstate=nstate)

    def test_chunk_size(self):
        """
        Check that the kinetic energy matrix does not depend on how many chunks
        the coordinate grid is split into.
        """
        for name, mol in tqdm(self.create_test_molecules().items()):
            with self.subTest(molecule=name):
                self.check_chunk_size(mol)

    def test_single_state_von_Weizsaecker(self):
        """
        Check that for a single electronic state the lower bound is equal to
        the von-Weizsäcker kinetic energy.
        """
        for name, mol in tqdm(
                self.create_test_molecules().items()):
            with self.subTest(molecule=name):
                self.check_single_state_von_Weizsaecker(mol)


if __name__ == "__main__":
    unittest.main(failfast=True)
