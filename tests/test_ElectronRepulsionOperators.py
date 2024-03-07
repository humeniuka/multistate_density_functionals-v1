#!/usr/bin/env python
# coding: utf-8
from abc import ABC, abstractmethod

import numpy
import numpy.linalg as la
import numpy.testing

import pyscf.dft
import pyscf.dft.libxc
import pyscf.fci
import pyscf.gto
import pyscf.scf

from tqdm import tqdm
import unittest

from msdft.ElectronRepulsionOperators import ExchangeCorrelationLikeFunctional
from msdft.ElectronRepulsionOperators import HartreeLikeFunctional
from msdft.ElectronRepulsionOperators import HartreeLikeFunctionalPoisson
from msdft.ElectronRepulsionOperators import LDACorrelationLikeFunctional
from msdft.ElectronRepulsionOperators import LDAExchangeLikeFunctional
from msdft.ElectronRepulsionOperators import LSDAExchangeLikeFunctional
from msdft.MultistateMatrixDensity import MultistateMatrixDensity
from msdft.MultistateMatrixDensity import MultistateMatrixDensityFCI


class TestHartreeLikeFunctional(unittest.TestCase):
    def create_test_molecules(self):
        """ dictionary with different molecules to run the tests on """
        molecules = {
            # 1-electron systems
            'hydrogen atom': pyscf.gto.M(
                atom = 'H 0 0 0',
                basis = '6-31g',
                # doublet
                spin = 1),
            # many electrons
            'water': pyscf.gto.M(
                atom = 'O  0 0 0; H 0.75 0.00 0.50; H 0.75 0.00 -0.50',
                basis = 'sto-3g',
                # singlet
                spin = 0),
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
        assert nstate > 0
        hf = pyscf.scf.RHF(mol)
        # supress printing of SCF energy
        hf.verbose = 0
        # compute self-consistent field
        hf.kernel()

        fci = pyscf.fci.FCI(mol, hf.mo_coeff)
        # Solve for one state more than requested to avoid
        # problems when nstate == 1.
        fci.nroots = nstate+1
        fci_energies, fcivecs = fci.kernel()
        # Remove the additional state again. For small basis sets,
        # there can be fewer states than requested.
        if len(fcivecs) == nstate+1:
            fcivecs = fcivecs[:-1]

        msmd = MultistateMatrixDensityFCI(mol, hf, fci, fcivecs)

        return msmd

    def check_exact_hartree_energy(self, mol, nstate=1):
        """
        The Hartree-like energy is calculated in two different ways:
         1) By applying get_jk(...) to each (transition) density and
            contracting the resulting electrostatic potential V(r) with
            the matrix density D(r).
         2) By contracting the electroc repulsion integrals (ab|cd) in the
            AO basis with the AO (transition) density matrices (reference).

        :param mol: A test molecule
        :type mol: gto.Mole

        :param nstate: Number of electronic states in the subspace.
           The full CI problem is solved for the lowest nstate states.
        :type nstate: int > 0
        """
        # functional for Hartree-like energy J[D(r)]
        hartree_like_functional = HartreeLikeFunctional(mol)

        # compute D(r) from full CI
        msmd = self.create_matrix_density(mol, nstate=nstate)

        # Evaluate J[D(r)] using J-build.
        J_msdft = hartree_like_functional(msmd)

        # The reference potential energy matrix is calculated by contracting the
        # (transition) density matrices in the AO basis with the electron
        # repulsion integrals.

        # Electron repulsion integrals (ab|cd)
        coulomb_integrals = msmd.exact_coulomb_energy()
        J_ref = 0.5 * numpy.einsum('ikkj->ij', coulomb_integrals)

        numpy.testing.assert_almost_equal(J_msdft, J_ref, decimal=3)

    def test_hartree_like_matrix(self):
        """
        Compare Hartree term J[D(r)] from numerical integration with exact matrix elements.
        """
        for name, mol in tqdm(
                self.create_test_molecules().items()):
            for nstate in tqdm([1,2]):
                with self.subTest(molecule=name, nstate=nstate):
                    self.check_exact_hartree_energy(mol, nstate=nstate)


class TestHartreeLikeFunctionalPoisson(unittest.TestCase):
    def create_test_molecules(self):
        """ dictionary with different molecules to run the tests on """
        molecules = {
            # 1-electron systems
            'hydrogen atom': pyscf.gto.M(
                atom = 'H 0 0 0',
                basis = '6-31g',
                # doublet
                spin = 1),
            # many electrons
            'water': pyscf.gto.M(
                atom = 'O  0 0 0; H 0.75 0.00 0.50; H 0.75 0.00 -0.50',
                basis = 'sto-3g',
                # singlet
                spin = 0),
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
        assert nstate > 0
        hf = pyscf.scf.RHF(mol)
        # supress printing of SCF energy
        hf.verbose = 0
        # compute self-consistent field
        hf.kernel()

        fci = pyscf.fci.FCI(mol, hf.mo_coeff)
        # Solve for one state more than requested to avoid
        # problems when nstate == 1.
        fci.nroots = nstate+1
        fci_energies, fcivecs = fci.kernel()
        # Remove the additional state again. For small basis sets,
        # there can be fewer states than requested.
        if len(fcivecs) == nstate+1:
            fcivecs = fcivecs[:-1]

        msmd = MultistateMatrixDensityFCI(mol, hf, fci, fcivecs)

        return msmd

    def check_exact_hartree_energy(self, mol, nstate=1):
        """
        The Hartree-like energy is calculated in two different ways:
         1) By solving the Poisson equation for each (transition) density and
            integrating the product of the resulting electrostatic potential
            V(r) with the matrix density D(r) numerically on a grid
         2) By contracting the electroc repulsion integrals (ab|cd) in the
            AO basis with the AO (transition) density matrices (exact).

        :param mol: A test molecule
        :type mol: gto.Mole

        :param nstate: Number of electronic states in the subspace.
           The full CI problem is solved for the lowest nstate states.
        :type nstate: int > 0
        """
        # functional for Hartree-like energy J[D(r)]
        hartree_like_functional = HartreeLikeFunctionalPoisson(mol)

        # compute D(r) from full CI
        msmd = self.create_matrix_density(mol, nstate=nstate)

        # Evaluate J[D(r)] by solving the Poisson equation and integration on the grid.
        J_msdft = hartree_like_functional(msmd)

        # The exact potential energy matrix is calculated by contracting the
        # (transition) density matrices in the AO basis with the electron
        # repulsion integrals.

        # Electron repulsion integrals (ab|cd)
        coulomb_integrals = msmd.exact_coulomb_energy()
        J_exact = 0.5 * numpy.einsum('ikkj->ij', coulomb_integrals)

        numpy.testing.assert_almost_equal(J_msdft, J_exact, decimal=3)

    def test_hartree_like_matrix(self):
        """
        Compare Hartree term J[D(r)] from numerical integration with exact matrix elements.
        """
        for name, mol in tqdm(
                self.create_test_molecules().items()):
            for nstate in tqdm([1,2]):
                with self.subTest(molecule=name, nstate=nstate):
                    self.check_exact_hartree_energy(mol, nstate=nstate)


class ExchangeCorrelationFunctionalTestCase(ABC, unittest.TestCase):
    """
    Abstract base class for all exchange/correlation energy functional tests.
    It contains functions needed by all tests.
    """
    @property
    @abstractmethod
    def xc_functional_class(self):
        """
        The subclass of :class:`~.ExchangeCorrelationLikeFunctional` for which
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
                basis = 'aug-cc-pvdz',
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

    def create_closed_shell_test_molecules(self):
        """ dictionary with closed-shell many-electron molecules to run the tests on """
        molecules = {
            # 4-electron system, closed shell
            'lithium hydride': pyscf.gto.M(
                atom = 'Li 0 0 0; H 0 0 1.60',
                basis = '6-31g',
                # singlet
                spin = 0),
            # many electrons
            'water': pyscf.gto.M(
                atom = 'O  0 0 0; H 0.75 0.00 0.50; H 0.75 0.00 -0.50',
                basis = 'sto-3g',
                # singlet
                spin = 0)
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
        assert nstate > 0
        hf = pyscf.scf.RHF(mol)
        # supress printing of SCF energy
        hf.verbose = 0
        # compute self-consistent field
        hf.kernel()

        fci = pyscf.fci.FCI(mol, hf.mo_coeff)
        # Solve for one state more than requested to avoid
        # problems when nstate == 1.
        fci.nroots = nstate+1
        fci_energies, fcivecs = fci.kernel()
        # Remove the additional state again. For small basis sets,
        # there can be fewer states than requested.
        if len(fcivecs) == nstate+1:
            fcivecs = fcivecs[:-1]

        msmd = MultistateMatrixDensityFCI(mol, hf, fci, fcivecs)

        return msmd

    def check_chunk_size(self, mol):
        """
        Compute exchange/correlation matrix with different chunk sizes.
        """
        # Check that the derived unit test is implemented correctly.
        assert issubclass(self.xc_functional_class, ExchangeCorrelationLikeFunctional)
        # functional for exchange/correlation part of electron-repulsion operator, XC[D(r)]
        exchange_correlation_functional = self.xc_functional_class(mol, level=1)

        msmd = self.create_matrix_density(mol, nstate=3)
        # XCij with default chunk size
        xc_matrix_ref = exchange_correlation_functional(msmd)
        # Increase the number of chunks by reducing the available memory
        # per chunk to 2**22 (~4 Mb) or 2**23 (~ 8Mb) bytes.
        for memory in [2**22, 2**23]:
            xc_matrix = exchange_correlation_functional(msmd, available_memory=memory)

            numpy.testing.assert_almost_equal(
                xc_matrix, xc_matrix_ref)


class LDAExchangeFunctionalSingleState(object):
    """
    Exchange energy in the local-density approximation for a closed-shell
    ground state density:

        Eₓ[ρ] = Cₓ ∫ ρ(r)⁴ᐟ³ dr
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
        Compute the exchange-energy for the density of a single electronic state.

        :param msmd: A multistate density matrix with only a single electronic state.
           The density should belong to a closed-shell electronic state.
        :type msmd: :class:`~.MultistateMatrixDensity`

        :return exchange_energy: A 1x1 matrix with the scalar exchange energy.
        :rtype exchange_energy: numpy.ndarray of shape (1,1)
        """
        # number of electronic states
        nstate = msmd.number_of_states
        assert nstate == 1, \
           "The LDA exchange functional is only defined for a single electronic state."

        # Evaluate D(r) on the integration grid.
        D, _, _ = msmd.evaluate(self.grids.coords)
        # Trace out spin and electronic states (there is only one state) to get ρ(r)
        rho = numpy.einsum('siir->r', D)

        # t[ρ] = Cₓ ρ(r)⁴ᐟ³
        exchange_energy_density = LSDAExchangeLikeFunctional.Cx * pow(rho, 4.0/3.0)

        # Integrate over space, Eₓ[ρ] = ∫ t[ρ] dr = Cₓ ∫ ρ(r)⁴ᐟ³ dr
        exchange_energy = numpy.einsum('r,r->', self.grids.weights, exchange_energy_density)

        # Reshape energy as a 1x1 matrix.
        exchange_matrix = numpy.array([[exchange_energy]])

        return exchange_matrix


class TestLSDAExchangeLikeFunctional(ExchangeCorrelationFunctionalTestCase):
    @property
    def xc_functional_class(self):
        """ The functional to be tested. """
        return LSDAExchangeLikeFunctional

    def test_chunk_size(self):
        """
        Check that the exchange energy matrix does not depend on how many chunks
        the coordinate grid is split into.
        """
        for name, mol in tqdm(self.create_closed_shell_test_molecules().items()):
            with self.subTest(molecule=name):
                self.check_chunk_size(mol)

    def test_local_density_exchange_functional(self):
        """
        Check that for a single closed-shell electronic state the multistate LSDA exchange energy
        functional agrees with the ground state LSDA functional.
        """
        for name, mol in tqdm(self.create_closed_shell_test_molecules().items()):
            with self.subTest(molecule=name):
                # scalar D(r) from single electronic state
                msmd = self.create_matrix_density(mol, nstate=1)

                # functionals for exchange energy, K[D(r)]
                exchange_functional_multi = self.xc_functional_class(mol)
                exchange_functional_single = LDAExchangeFunctionalSingleState(mol)

                # Compare the multistate and the single-state exchange functionals.
                exchange_matrix_multi = exchange_functional_multi(msmd)
                exchange_matrix_single = exchange_functional_single(msmd)

                numpy.testing.assert_almost_equal(
                    exchange_matrix_single, exchange_matrix_multi)


class TestLDAExchangeLikeFunctional(ExchangeCorrelationFunctionalTestCase):
    @property
    def xc_functional_class(self):
        """ The functional to be tested. """
        return LDAExchangeLikeFunctional

    def test_chunk_size(self):
        """
        Check that the exchange energy matrix does not depend on how many chunks
        the coordinate grid is split into.
        """
        for name, mol in tqdm(self.create_closed_shell_test_molecules().items()):
            with self.subTest(molecule=name):
                self.check_chunk_size(mol)

    def test_local_density_exchange_functional(self):
        """
        Check that for a single closed-shell electronic state the multistate LDA exchange energy
        functional agrees with the ground state LDA functional.
        """
        for name, mol in tqdm(self.create_closed_shell_test_molecules().items()):
            with self.subTest(molecule=name):
                # scalar D(r) from single electronic state
                msmd = self.create_matrix_density(mol, nstate=1)

                # functionals for exchange energy, K[D(r)]
                exchange_functional_multi = self.xc_functional_class(mol)
                exchange_functional_single = LDAExchangeFunctionalSingleState(mol)

                # Compare the multistate and the single-state exchange functionals.
                exchange_matrix_multi = exchange_functional_multi(msmd)
                exchange_matrix_single = exchange_functional_single(msmd)

                numpy.testing.assert_almost_equal(
                    exchange_matrix_single, exchange_matrix_multi)


class TestLDACorrelationLikeFunctional(ExchangeCorrelationFunctionalTestCase):
    @property
    def xc_functional_class(self):
        """ The functional to be tested. """
        return LDACorrelationLikeFunctional

    def test_chunk_size(self):
        """
        Check that the correlation energy matrix does not depend on how many chunks
        the coordinate grid is split into.
        """
        for name, mol in tqdm(self.create_closed_shell_test_molecules().items()):
            with self.subTest(molecule=name):
                self.check_chunk_size(mol)

    def test_chachiyo_functional_implementation(self):
        """
        Check that the implementation of the correlation functional gives the same energy
        as libxc for a range of electron densities.
        """
        for name, mol in tqdm(self.create_closed_shell_test_molecules().items()):
            with self.subTest(molecule=name):
                # scalar D(r) from single electronic state
                msmd = self.create_matrix_density(mol, nstate=1)

                # Evaluate the correlation energy, X[D(r)], using the multi-state functional.
                correlation_functional_multi = self.xc_functional_class(mol)
                correlation_matrix_multi = correlation_functional_multi(msmd)

                # Evaluate the LDA correlation energy of a single state using the
                # implementation of libxc.
                grids = pyscf.dft.gen_grid.Grids(mol)
                grids.level = 8
                grids.build()
                # number of grid points
                ncoord = grids.coords.shape[0]

                D, _, _ = msmd.evaluate(grids.coords)
                # rho (*,N) are ordered as (den,grad_x,grad_y,grad_z,laplacian,tau)
                # but only den has to be non-zero for an LDA functional.
                rho = numpy.zeros((6, ncoord))
                # Sum over spins.
                rho[0,:] = D[0,0,0,:] + D[1,0,0,:]
                exc, _, _, _ = pyscf.dft.libxc.eval_xc(',LDA_C_CHACHIYO', rho)
                # Integrate over space
                #  Ec[ρ] = ∫ ρ exc[ρ] dr
                correlation_matrix_single = numpy.array([[
                        numpy.sum(grids.weights * (rho * exc))
                    ]])

                # Compare the multistate and the single-state (libxc) correlation functionals.
                numpy.testing.assert_almost_equal(
                    correlation_matrix_single, correlation_matrix_multi, decimal=5)


if __name__ == "__main__":
    unittest.main()
