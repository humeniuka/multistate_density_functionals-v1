#!/usr/bin/env python
# coding: utf-8
from abc import ABC, abstractmethod
import numpy
import pyscf.gto

from tqdm import tqdm
import unittest

from msdft.ElectronRepulsionOperators import HartreeLikeFunctionalPoisson
from msdft.ElectronRepulsionOperators import LSDAExchangeLikeFunctional
from msdft.LowerBoundElectronRepulsion import LiebOxfordBound
from msdft.LowerBoundElectronRepulsion import LowerBoundElectronRepulsionSubspaceInvariant
from msdft.LowerBoundElectronRepulsion import LowerBoundElectronRepulsionSumOverStates
from msdft.MultistateMatrixDensity import MultistateMatrixDensity
from msdft.MultistateMatrixDensity import MultistateMatrixDensityFCI

from test_ElectronRepulsionOperators import LDAExchangeFunctionalSingleState


def create_density_function(msmd: MultistateMatrixDensity) -> callable:
    """
    return a function that evaluates the ground state density on a grid.

    :param msmd: The multistate matrix density for a single state.
    :type msmd: :class:`~.MultistateMatrixDensity`

    :return: density function `density_function(x,y,z)`
    :rtype: callable
    """
    assert msmd.number_of_states == 1, "Only the ground state density is evaluated."
    # A function for evaluating the total density on a multicenter Becke grid.
    def density_function(x, y, z):
        # The `becke` module and the `msdft` module use different
        # shapes for the coordinates of the grid points. Before
        # passing the grid to `msdft`, the arrays have to be flattened
        # and reshaped as (ncoord,3). Before returning the density,
        # it has to be brought into the same shape as each input coordinate
        # array.
        coords = numpy.vstack(
            [x.flatten(), y.flatten(), z.flatten()]).transpose()

        # Evaluate the density.
        spin_density, _, _ = msmd.evaluate(coords)
        # The Coulomb potential does not distinguish spins, so
        # sum over spins.
        density = spin_density[0,0,0,:] + spin_density[1,0,0,:]

        # Give it the same shape as the input arrays.
        density = numpy.reshape(density, x.shape)
        return density

    return density_function


class TestLiebOxfordBound(unittest.TestCase):
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

    def check_direct_Coulomb_energy(self, mol):
        """
        The direct Coulomb energy of the LiebOxfordBound object is compared
        the Hartree energy of the HartreeLikeFunctional.
        """
        # Solve the electronic structure for the ground state with full CI.
        msmd = MultistateMatrixDensityFCI.create_matrix_density(mol, nstate=1)
        # The reference Hartree energy.
        hartree_functional = HartreeLikeFunctionalPoisson(mol)
        hartree_energy_ref = hartree_functional(msmd)[0,0]

        lieb_oxford_bound = LiebOxfordBound(mol)
        # function for evaluating ground state density
        density_function = create_density_function(msmd)
        # compute 1/2 ∫∫ ρ(r) ρ(r') / |r-r'|
        hartree_energy = lieb_oxford_bound.direct_coulomb_energy(density_function)

        self.assertAlmostEqual(hartree_energy_ref, hartree_energy)

    def test_direct_Coulomb_energy(self):
        """ check that the direct Coulomb energy is equal to the Hartree term """
        for name, mol in tqdm(self.create_test_molecules().items()):
            with self.subTest(molecule=name):
                self.check_direct_Coulomb_energy(mol)

    def check_bound_on_indirect_energy(self, mol):
        """
        The lower bound on the indirect Coulomb energy is a multiple of the
        LSDA exchange energy,

            Eₓ[ρ] = Cₓ ∫ ρ(r)⁴ᐟ³ dr

        so

            Lieb-Oxford bound = - cᴸᴼ/Cₓ Eₓ[ρ]
        """
        # Solve the electronic structure for the ground state with full CI.
        msmd = MultistateMatrixDensityFCI.create_matrix_density(mol, nstate=1)
        # The reference LSDA exchange energy.
        exchange_functional = LDAExchangeFunctionalSingleState(mol)
        exchange_energy = exchange_functional(msmd)[0,0]

        lieb_oxford_bound = LiebOxfordBound(mol)
        # function for evaluating ground state density
        density_function = create_density_function(msmd)
        # compute  cᴸᴼ ∫ ρ(r)⁴ᐟ³
        lower_bound_indirect = lieb_oxford_bound.bound_on_indirect_energy(density_function)

        C_LO = LiebOxfordBound.Lieb_Oxford_constant
        Cx = LSDAExchangeLikeFunctional.Cx
        self.assertAlmostEqual(-C_LO/Cx * exchange_energy, lower_bound_indirect)

    def test_bound_on_indirect_energy(self):
        """ check that the lower bound on the indirect part of the Coulomb energy """
        for name, mol in tqdm(self.create_test_molecules().items()):
            with self.subTest(molecule=name):
                self.check_bound_on_indirect_energy(mol)

    def check_lower_bound(self, mol):
        """
        Verify that the expectation value of the electron repulsion is bounded from
        below by the Lieb-Oxford bound.
        """
        # Solve the electronic structure for the ground state with full CI.
        msmd = MultistateMatrixDensityFCI.create_matrix_density(mol, nstate=1)
        # Compute <Ψ|∑ᵦ<ᵧ 1/|rᵦ-rᵧ||Ψ>
        exact_electron_repulsion = msmd.exact_electron_repulsion()[0,0]

        lieb_oxford_bound = LiebOxfordBound(mol)
        # function for evaluating ground state density
        density_function = create_density_function(msmd)
        # compute 1/2 ∫∫ ρ(r) ρ(r') / |r-r'|  -  cᴸᴼ ∫ ρ(r)⁴ᐟ³
        lower_bound = lieb_oxford_bound(density_function)

        # Compare exact electron repulsion with Lieb-Oxford bound.
        self.assertLessEqual(lower_bound, exact_electron_repulsion)

    def test_lower_bound(self):
        """
        Verify that the expectation value of the electron repulsion is bounded from
        below by the Lieb-Oxford bound.
        """
        for name, mol in tqdm(self.create_test_molecules().items()):
            with self.subTest(molecule=name):
                self.check_lower_bound(mol)


class LowerBoundElectronRepulsionTestCase(ABC, unittest.TestCase):
    """
    Abstract base class for testing the electron repulsion energy bounds.
    It contains functions needed by all tests.
    """
    @property
    @abstractmethod
    def lower_bound_electron_repulsion_class(self):
        """
        The subclass of :class:`~.LowerBoundElectronRepulsion` for which
        the respective unit test is written.
        """
        pass

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

    def check_single_state(self, mol):
        """
        For a single electronic state the bound should be equal to the
        Lieb-Oxford bound.
        """
        # Solve the electronic structure for the ground state with full CI.
        msmd = MultistateMatrixDensityFCI.create_matrix_density(mol, nstate=1)

        # For a single state, these bound should be identical
        lieb_oxford_functional = LiebOxfordBound(mol)
        # function for evaluating ground state density
        density_function = create_density_function(msmd)
        lieb_oxford_bound = lieb_oxford_functional(density_function)

        # Lower bound that should be tested.
        lower_bound_functional = self.lower_bound_electron_repulsion_class()(mol)
        lower_bound = lower_bound_functional(msmd)

        self.assertAlmostEqual(lieb_oxford_bound, lower_bound)

    def check_lower_bound(self, mol, nstate=4):
        """
        Verify that the state-average expectation value of the electron repulsion
        is bounded from below.
        """
        # Solve the electronic structure for the ground state with full CI.
        msmd = MultistateMatrixDensityFCI.create_matrix_density(mol, nstate=nstate)
        # Number of electronic states (could be smaller than the keyword `nstate`)
        nstate = msmd.number_of_states
        # Compute 1/N ∑ᵢ <Ψᵢ|∑ᵦ<ᵧ 1/|rᵦ-rᵧ||Ψᵢ>
        subspace_electron_repulsion = 1.0/nstate * numpy.trace(msmd.exact_electron_repulsion())

        # Lower bound on subspace electron repulsion.
        lower_bound_functional = self.lower_bound_electron_repulsion_class()(mol)
        lower_bound = lower_bound_functional(msmd)

        # Compare exact electron repulsion with the lower bound.
        self.assertLessEqual(lower_bound, subspace_electron_repulsion)


class TestLowerBoundElectronRepulsionSumOverStates(LowerBoundElectronRepulsionTestCase):
    def lower_bound_electron_repulsion_class(self):
        return LowerBoundElectronRepulsionSumOverStates

    def test_single_state(self):
        """
        Verify that for a single state the bound reduces to the familiar Lieb-Oxford bound.
        """
        for name, mol in tqdm(self.create_test_molecules().items()):
            with self.subTest(molecule=name):
                self.check_single_state(mol)

    def test_lower_bound(self):
        """
        Verify that the subspace electron repulsion is larger than the bound.
        """
        for name, mol in tqdm(self.create_test_molecules().items()):
            for nstate in tqdm([2,3]):
                with self.subTest(molecule=name, nstate=nstate):
                    self.check_lower_bound(mol, nstate=nstate)


class TestLowerBoundElectronRepulsionSubspaceInvariant(LowerBoundElectronRepulsionTestCase):
    def lower_bound_electron_repulsion_class(self):
        return LowerBoundElectronRepulsionSubspaceInvariant

    def test_single_state(self):
        """
        Verify that for a single state the bound reduces to the familiar Lieb-Oxford bound.
        """
        for name, mol in tqdm(self.create_test_molecules().items()):
            with self.subTest(molecule=name):
                self.check_single_state(mol)

    def test_lower_bound(self):
        """
        Verify that the subspace electron repulsion is larger than the bound.
        """
        for name, mol in tqdm(self.create_test_molecules().items()):
            for nstate in tqdm([2,3]):
                with self.subTest(molecule=name, nstate=nstate):
                    self.check_lower_bound(mol, nstate=nstate)


if __name__ == "__main__":
    unittest.main(failfast=True)
