#!/usr/bin/env python
# coding: utf-8
from abc import ABC, abstractmethod

import numpy
import numpy.linalg as la
import numpy.testing

import pyscf.dft
import pyscf.fci
import pyscf.gto
import pyscf.scf
import pyscf.tddft

from tqdm import tqdm
import unittest

from msdft.MultistateMatrixDensity import MultistateMatrixDensityCASCI
from msdft.MultistateMatrixDensity import MultistateMatrixDensityCASSCF
from msdft.MultistateMatrixDensity import MultistateMatrixDensityCISD
from msdft.MultistateMatrixDensity import MultistateMatrixDensityFCI
from msdft.MultistateMatrixDensity import MultistateMatrixDensityTDDFT


class BaseTestMultistateMatrixDensity(ABC):
    @abstractmethod
    def create_test_molecules(self):
        """ dictionary with different molecules to run the tests on """
        pass

    @abstractmethod
    def create_matrix_density(self, mol, nstate=4):
        """
        Compute multistate matrix density for the lowest few excited states
        of a small molecule using the electronic structure method of the
        derived MultistateMatrixDensity class that is to be tested.

        :param mol: A test molecule
        :type mol: gto.Mole

        :param nstate: number of excited states to calculate
        :type nstate: positive int

        :return: multistate matrix density
        :rtype: MultistateMatrixDensity
        """
        pass

    def check_integrals(self, msmd):
        """
        check that the state density integrates to the correct number of electrons
        and that the transition density integrates to 0.

        :param msmd: The multistate matrix density to be tested.
        :type msmd: MultistateMatrixDensity
        """
        # integration grid
        grids = pyscf.dft.gen_grid.Grids(msmd.mol)
        grids.level = 8
        grids.build()

        D, grad_D, lapl_D = msmd.evaluate(grids.coords)
        trace_D = numpy.einsum('siir->sr', D)

        # The integral also involves a sum over spins.
        integrals = numpy.einsum('r,sijr->ij', grids.weights, D)

        nstate = msmd.number_of_states
        number_of_electrons = sum(msmd.mol.nelec)
        for i in range(0, nstate):
            for j in range(0, nstate):
                with self.subTest(i=i, j=j):
                    if i == j:
                        # State densities should integrate to the number of electrons.
                        self.assertAlmostEqual(number_of_electrons, integrals[i,i], places=3)
                    else:
                        # Integrating the transition density, just gives the overlap between
                        # the states, which should be zero for different eigenstates.
                        self.assertAlmostEqual(0.0, integrals[i,j])

        # The trace over spin and electronic states should be equal to
        # (number of electrons) x (number of states)
        integral_trace_D = numpy.einsum('r,sr->', grids.weights, trace_D)
        self.assertAlmostEqual(integral_trace_D, number_of_electrons*nstate)

    def test_integrals(self):
        """ Check integrals of D(r) for all test molecules """
        for name, mol in tqdm(self.create_test_molecules().items()):
            with self.subTest(molecule=name):
                # Example density.
                msmd = self.create_matrix_density(mol)
                self.check_integrals(msmd)

    def check_gradient_and_laplacian(self, mol):
        """
        compare analytical gradients ∇D(r) and ∇tr(D)(r) and the Laplacian ∇²D(r)
        with numerical ones from finite differences
        """
        # Example density.
        msmd = self.create_matrix_density(mol)
        # Gradients are checked at random coordinates.
        ncoord = 100
        # The hardcoded seed ensures that the same random numbers are used
        # every time the test is run. Otherwise the test fails occasionally
        # when the threshold is is too tight.
        random_number_generator = numpy.random.default_rng(seed=2345)
        coords = 5.0*(random_number_generator.random((ncoord,3)) - 0.5)

        # Analytical gradients and Laplacian of D
        D, grad_D, lapl_D = msmd.evaluate(coords)

        # Trace out electronic states to get tr(D) and ∇tr(D)
        trace_D = numpy.einsum('siir->sr', D)
        grad_trace_D = numpy.einsum('siiar->sar', grad_D)

        # Numerical gradients of D and tr(D)
        grad_D_numerical = numpy.zeros_like(grad_D)
        lapl_D_numerical = numpy.zeros_like(lapl_D)
        grad_trace_D_numerical = numpy.zeros_like(grad_trace_D)

        # dD/dx = [D(x+h) - D(x-h)]/(2 h)
        h = 0.001
        for xyz in [0,1,2]:
            # unit vector in the x,y or z-direction
            unit_vector = numpy.zeros(3)
            unit_vector[xyz] = 1.0

            # D(r+h*e_x)
            D_plus, _, _ = msmd.evaluate(coords + h*unit_vector)
            trace_D_plus = numpy.einsum('siir->sr', D_plus)
            # D(r-h*e_x)
            D_minus, _, _ = msmd.evaluate(coords - h*unit_vector)
            trace_D_minus = numpy.einsum('siir->sr', D_minus)

            # finite difference gradient
            grad_D_numerical[:,:,:,xyz,:] = (D_plus - D_minus)/(2*h)
            grad_trace_D_numerical[:,xyz,:] = (trace_D_plus - trace_D_minus)/(2*h)

            # Add finite difference approximation for second derivative to
            # numerical Laplacian.
            lapl_D_numerical += (D_plus - 2*D + D_minus)/pow(h,2)

        # Compare analytical and numerical gradients
        with self.subTest("gradient of D(r)"):
            # relative error |∇D-∇D(numerical)|/|∇D(numerical)|
            relative_error = (
                la.norm(grad_D - grad_D_numerical)/la.norm(grad_D_numerical))
            self.assertLess(relative_error, 1.0e-3)
            numpy.testing.assert_almost_equal(grad_D, grad_D_numerical, decimal=2)
        with self.subTest("gradient of tr(D)"):
            # relative error |∇trD-∇trD(numerical)|/|∇trD(numerical)|
            relative_error = (
                la.norm(grad_trace_D - grad_trace_D_numerical)/la.norm(grad_trace_D_numerical))
            self.assertLess(relative_error, 1.0e-3)
            numpy.testing.assert_almost_equal(grad_trace_D, grad_trace_D_numerical, decimal=2)
        with self.subTest("Laplacian of D(r)"):
            # relative error |∇²D-∇²D(numerical)|/|∇²D(numerical)|
            relative_error = (
                la.norm(lapl_D - lapl_D_numerical)/la.norm(lapl_D_numerical))
            self.assertLess(relative_error, 1.0e-3)
            numpy.testing.assert_almost_equal(lapl_D, lapl_D_numerical, decimal=2)

    def test_gradient_and_laplacian(self):
        """ Compare numerical and analytical gradient ∇D(r) and Laplacian ∇²D(r) for all test molecules """
        for name, mol in tqdm(self.create_test_molecules().items()):
            with self.subTest(molecule=name):
                self.check_gradient_and_laplacian(mol)

    def check_derivatives(self, mol, deriv=2):
        """
        compare analytical derivatives ∂ⁿ/∂xⁿ D(x,y,z), ∂ⁿ/∂yⁿ D(x,y,z) and ∂ⁿ/∂zⁿ D(x,y,z)
        for n=1,2,..,deriv with numerical ones from finite differences

        :param mol: test molecule that provides the matrix density
        :type mol: pyscf.gto.Mole

        :param deriv: maximum order of derivatives
        :type deriv: int >= 0
        """
        # Example density.
        msmd = self.create_matrix_density(mol)
        # Derivatives are checked at random coordinates.
        ncoord = 100
        # The hardcoded seed ensures that the same random numbers are used
        # every time the test is run.
        random_number_generator = numpy.random.default_rng(seed=1234)
        coords = 5.0*(random_number_generator.random((ncoord,3)) - 0.5)

        # Analytical derivatives
        D_derivs = msmd.evaluate_derivatives(coords, deriv=deriv)

        # Numerical derivatives of D
        D_derivs_numerical = numpy.zeros_like(D_derivs)

        # The numerical derivatives of order n are computed from the finite
        # difference quotient of the derivatives of order n-1,
        #   ∂ⁿ/∂xⁿ D(x) = [∂ⁿ⁻¹/∂xⁿ⁻¹ D(x+h) - ∂ⁿ⁻¹/∂xⁿ⁻¹ D(x-h)]/(2h)

        # Step size for finite difference quotient.
        h = 0.001
        for xyz in [0,1,2]:
            # Unit vector in the x,y or z-direction.
            unit_vector = numpy.zeros(3)
            unit_vector[xyz] = 1.0

            # ∂ⁿ/∂xⁿ D(x+h)
            D_derivs_plus = msmd.evaluate_derivatives(coords + h*unit_vector, deriv=deriv)
            # ∂ⁿ/∂xⁿ D(x-h)
            D_derivs_minus = msmd.evaluate_derivatives(coords - h*unit_vector, deriv=deriv)

            # finite difference quotients
            for n in range(1, deriv+1):
                # ∂ⁿ/∂xⁿ D(x) = [∂ⁿ⁻¹/∂xⁿ⁻¹ D(x+h) - ∂ⁿ⁻¹/∂xⁿ⁻¹ D(x-h)]/(2h)
                D_derivs_numerical[:,:,:,xyz,n,:] = (
                    D_derivs_plus[:,:,:,xyz,n-1,:] - D_derivs_minus[:,:,:,xyz,n-1,:])/(2*h)

        # Compare the matrix density.
        D, _, _ = msmd.evaluate(coords)
        for xyz in [0,1,2]:
            # The 0-th derivatives ∂⁰/∂x⁰ D(x), ∂⁰/∂y⁰ D(x), ∂⁰/∂z⁰ D(x) are just equal to D(r).
            numpy.testing.assert_almost_equal(D, D_derivs[:,:,:,xyz,0,:])

        # Compare analytical and numerical derivatives of order n=1,...,deriv
        for n in range(1, deriv+1):
            for xyz in [0,1,2]:
                with self.subTest(f"derivative ∂ⁿ/∂xyzⁿ D(x) of order n={n} xyz={xyz} (0-x, 1-y, 2-z)"):
                    numpy.testing.assert_almost_equal(
                        D_derivs_numerical[:,:,:,:,n,:], D_derivs[:,:,:,:,n,:], decimal=4)
                    # relative error |∂ⁿ/∂xⁿ D(x) - ∂ⁿ/∂xⁿ D(x)(numerical)|/|∂ⁿ/∂xⁿ D(x)(numerical)|
                    relative_error = (
                        la.norm(D_derivs_numerical[:,:,:,:,n,:] - D_derivs[:,:,:,:,n,:]) /
                        la.norm(D_derivs_numerical[:,:,:,:,n,:])
                    )
                    self.assertLess(relative_error, 1.0e-4)

    def test_derivatives(self):
        """ Compare numerical and analytical derivatives of D(r) for all test molecules """
        for name, mol in tqdm(self.create_test_molecules().items()):
            with self.subTest(molecule=name):
                self.check_derivatives(mol)

    def check_kinetic_energy_density(self, mol):
        """
        Check that the kinetic energy density integrates
        to the correct kinetic energy.
        """
        msmd = self.create_matrix_density(mol)
        # Compute the kinetic energy matrix exactly
        kinetic_matrix_exact = msmd.exact_kinetic_energy()

        # Generate the multicenter integration grid.
        grids = pyscf.dft.gen_grid.Grids(mol)
        grids.level = 8
        grids.build()

        # Evaluate the two types of kinetic energy densities on the grid.
        T_lap, T_gg = msmd.kinetic_energy_density(grids.coords)
        # Integrate over spin and space, Tᵢⱼ = ∫ Tᵢⱼ(r) dr
        kinetic_matrix_lap = numpy.einsum('r,sijr->ij', grids.weights, T_lap)
        kinetic_matrix_gg = numpy.einsum('r,sijr->ij', grids.weights, T_gg)

        # Compare with the exact matrix elements
        numpy.testing.assert_almost_equal(
            kinetic_matrix_gg, kinetic_matrix_exact, decimal=6)
        numpy.testing.assert_almost_equal(
            kinetic_matrix_lap, kinetic_matrix_exact, decimal=6)

    def test_kinetic_energy_density(self):
        """ compare integral of kinetic energy density with exact matrix elements """
        for name, mol in tqdm(self.create_test_molecules().items()):
            with self.subTest(molecule=name):
                self.check_kinetic_energy_density(mol)

    def check_align_phases(self, mol):
        """
        Check that the arbitrary global phases of the eigenfunctions can be removed
        by aligning with a reference.
        """
        # The reference D' is obtained by solving the RHF and Full CI and eigenvalue problems.
        msmd_ref = self.create_matrix_density(mol)
        # Solving the same eigenvalue problem again, might give the same or different global
        # phases in D as in D'.
        msmd = self.create_matrix_density(mol)
        # To be sure we have different signs, the density matrices are multiplied
        # by some random signs.
        signs = numpy.sign(numpy.random.rand(msmd.number_of_states)-0.5).astype(int)
        msmd.density_matrices = numpy.einsum('i,j,sijab->sijab', signs, signs, msmd.density_matrices)

        # After aligning the phases with the reference,
        # the matrix densities should be the same again.
        msmd.align_phases(msmd_ref)

        # For comparison, the matrix densities are evaluated on a coarse grid.
        # The density matrices in the AO basis might still differ in some irrelevant
        # signs, therefore it is better to compare D and D' on a grid.
        grids = pyscf.dft.gen_grid.Grids(mol)
        grids.level = 1
        grids.build()

        # D'ᵢⱼ(r)
        D_ref, _, _ = msmd_ref.evaluate(grids.coords)
        # σᵢσⱼ Dᵢⱼ(r), i.e. Dᵢⱼ(r) after aligning the phases with D'ᵢⱼ(r)
        D_aligned, _, _ = msmd_ref.evaluate(grids.coords)
        numpy.testing.assert_almost_equal(D_ref, D_aligned)

    def test_align_phases(self):
        """ Check that global phases can be found and removed. """
        for name, mol in tqdm(self.create_test_molecules().items()):
            with self.subTest(molecule=name):
                self.check_align_phases(mol)


class TestMultistateMatrixDensityFCI(BaseTestMultistateMatrixDensity, unittest.TestCase):
    def create_test_molecules(self):
        """ dictionary with different molecules to run the tests on """
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
                spin = 1),
            # 2-electron systems, paired spins
            'hydrogen molecule': pyscf.gto.M(
                atom = 'H 0 0 0; H 0 0 0.74',
                basis = '6-31g',
                charge = 0,
                spin = 0),
            # 2-electron systems, parallel spins
            'hydrogen molecule (triplet)': pyscf.gto.M(
                atom = 'H 0 0 0; H 0 0 0.74',
                basis = '6-31g',
                charge = 0,
                spin = 2),
            # 3-electron systems, one unpaired spin
            'lithium atom': pyscf.gto.M(
                atom = 'Li 0 0 0',
                basis = '6-31g',
                # doublet
                spin = 1),
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
                spin = 0),
            # effective core potential which removes the 1s orbital of oxygen
            'oxygen (ECP)': pyscf.gto.M(
                atom = 'O  0 0 0',
                basis = {'O': 'crenbl'},
                ecp = {'O': 'crenbl'},
                # triplet
                spin = 2),
        }
        return molecules

    def create_matrix_density(self, mol, nstate=4):
        # call the static method
        return MultistateMatrixDensityFCI.create_matrix_density(
            mol, nstate=nstate, spin_symmetry=False, raise_error=False)

    def check_hartree_matrix_product(self, mol, nstate=1):
        """
        The Hartree-like energy is calculated in two different ways:
         1) Using get_jk(...) to first compute the electrostatic potential of D(r)
            and then contracting with D(r). This does not require keeping all
            electron integrals in memory.
         2) By contracting the electron repulsion integrals (ab|cd) in the
            AO basis with the AO (transition) density matrices (exact).

        :param mol: A test molecule
        :type mol: gto.Mole

        :param nstate: Number of electronic states in the subspace.
           The full CI problem is solved for the lowest nstate states.
        :type nstate: int > 0
        """
        # compute D(r) from full CI
        msmd = self.create_matrix_density(mol, nstate=nstate)

        # Evaluate J[D(r)] using J-build.
        J_msdft = msmd.hartree_matrix_product()

        # The exact potential energy matrix is calculated by contracting the
        # (transition) density matrices in the AO basis with the electron
        # repulsion integrals.

        # Electron repulsion integrals (ab|cd)
        coulomb_integrals = msmd.exact_coulomb_energy()
        J_exact = 0.5 * numpy.einsum('ikkj->ij', coulomb_integrals)

        numpy.testing.assert_almost_equal(J_msdft, J_exact)

    def test_hartree_matrix_product(self):
        """
        Compare Hartree matrix product J[D(r)] from J-build with exact matrix elements
        """
        for name, mol in tqdm(
                self.create_test_molecules().items()):
            for nstate in tqdm([1,2]):
                with self.subTest(molecule=name, nstate=nstate):
                    self.check_hartree_matrix_product(mol, nstate=nstate)

    def test_exact_electron_repulsion(self):
        """
        If there is only a single electron, the matrix elements for the
        electron-electron repulsion operator should be zero.
        """
        # Hydrogen molecular ion.
        mol = pyscf.gto.M(
            atom = 'H 0 0 0; H 0 0 0.74',
            basis = '6-31g',
            charge = 1,
            spin = 1)
        msmd = self.create_matrix_density(mol)
        # electron-electron repulsion
        repulsion_matrix = msmd.exact_electron_repulsion()
        # No electron-electron repulsion.
        numpy.testing.assert_almost_equal(numpy.zeros_like(repulsion_matrix), repulsion_matrix)

    def test_create_matrix_density(self):
        """ check that matrix densities can be created for 1 or more states """
        for name, mol in tqdm(self.create_test_molecules().items()):
            for nstate in [1,2]:
                with self.subTest(molecule=name, nstate=nstate):
                    self.create_matrix_density(mol, nstate=nstate)


class TestMultistateMatrixDensityCISD(TestMultistateMatrixDensityFCI):
    def create_matrix_density(self, mol, nstate=4):
        # call the static method
        return MultistateMatrixDensityCISD.create_matrix_density(
            mol, nstate=nstate, raise_error=False)

    def compare_cisd_and_fci(self, mol, nstate=4):
        """
        For one- and two-electron systems CISD and FCI should produce exactly
        the same matrix densities (up to random global phases)
        """
        # Compute D(r) with full CI
        msmd_fci = MultistateMatrixDensityFCI.create_matrix_density(
            mol, nstate=nstate, spin_symmetry=True, raise_error=False)
        # Compute D(r) with CISD
        msmd_cisd = MultistateMatrixDensityCISD.create_matrix_density(
            mol, nstate=nstate, raise_error=False)
        # Remove differing global phases
        msmd_cisd.align_phases(msmd_fci)

        # The matrix densities are compared at random coordinates.
        ncoord = 100
        # The hardcoded seed ensures that the same random numbers are used
        # every time the test is run. Otherwise the test fails occasionally
        # when the threshold is is too tight.
        random_number_generator = numpy.random.default_rng(seed=6789)
        coords = 5.0*(random_number_generator.random((ncoord,3)) - 0.5)

        # Evalute D(r) on the grid.
        D_fci, _, _ = msmd_fci.evaluate(coords)
        D_cisd, _, _ = msmd_cisd.evaluate(coords)
        numpy.testing.assert_almost_equal(D_fci, D_cisd)

        # Eigenenergies should also be the same.
        numpy.testing.assert_almost_equal(msmd_fci.eigenenergies, msmd_cisd.eigenenergies)

    def test_cisd_versus_fci(self):
        """
        Check that matrix densities of 1- and 2-electron systems agree between FCI and CISD.
        """
        # Select some test molecules which have at most 2 electrons.
        # Atoms are also excluded, since they have spherical symmetry: The eigensolvers
        # will produce random linear combinations of degenerate states, so that it is
        # impossible to compare the matrix densities between FCI and CISD.
        test_molecules = self.create_test_molecules()
        test_molecule_1e_and_2e = {name: test_molecules[name]
            for name in [
                'hydrogen molecular ion',
                'hydrogen molecule',
                'hydrogen molecule (triplet)'
            ]
        }
        for name, mol in tqdm(test_molecule_1e_and_2e.items()):
            for nstate in [1,2]:
                with self.subTest(molecule=name, nstate=nstate):
                    self.compare_cisd_and_fci(mol, nstate=nstate)


class BaseTestMultistateMatrixDensityCAS(ABC):
    """ Common parts for tests of CASSCF and CASCI """
    def create_test_molecules(self):
        """ dictionary with different molecules to run the tests on """
        # Atoms are excluded, since they have spherical symmetry: The eigensolvers
        # will produce random linear combinations of degenerate states, so that it is
        # impossible to compare the matrix densities between FCI and CASSCF/CASCI.
        molecules = {
            'hydrogen molecular ion': pyscf.gto.M(
                atom = 'H 0 0 0; H 0 0 0.74',
                basis = '6-31g',
                charge = 1,
                spin = 1),
            # 2-electron systems, paired spins
            'hydrogen molecule': pyscf.gto.M(
                atom = 'H 0 0 0; H 0 0 0.74',
                basis = '6-31g',
                charge = 0,
                spin = 0),
            # 2-electron systems, parallel spins
            'hydrogen molecule (triplet)': pyscf.gto.M(
                atom = 'H 0 0 0; H 0 0 0.74',
                basis = '6-31g',
                charge = 0,
                spin = 2),
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
                spin = 0),
        }
        return molecules

    @abstractmethod
    def create_matrix_density(
            self,
            mol,
            nstate=2, ncas=None, nelecas=None,
            spin_symmetry=True, raise_error=True):
        # This returns either the CASSCF of the CASCI matrix density.
        pass

    def compare_cas_and_fci(self, mol, nstate=2, ncas=None, nelecas=None, decimal=6):
        """
        If the active space includes all orbitals and electrons,
        CASSCF/CASCI and FCI should produce exactly the same matrix densities
        (up to random global phases).
        """
        # Compute D(r) with full CI
        msmd_fci = MultistateMatrixDensityFCI.create_matrix_density(
            mol, nstate=nstate, spin_symmetry=True, raise_error=False)
        # Compute D(r) with CASSCF and full active space
        msmd_cas = self.create_matrix_density(
            mol,
            nstate=nstate, ncas=ncas, nelecas=nelecas,
            spin_symmetry=True, raise_error=False)
        # Remove differing global phases
        msmd_cas.align_phases(msmd_fci)

        # The matrix densities are compared at random coordinates.
        ncoord = 100
        # The hardcoded seed ensures that the same random numbers are used
        # every time the test is run. Otherwise the test fails occasionally
        # when the threshold is is too tight.
        random_number_generator = numpy.random.default_rng(seed=6789)
        coords = 5.0*(random_number_generator.random((ncoord,3)) - 0.5)

        # Evalute D(r) on the grid.
        D_fci, _, _ = msmd_fci.evaluate(coords)
        D_cas, _, _ = msmd_cas.evaluate(coords)
        numpy.testing.assert_almost_equal(D_fci, D_cas, decimal=decimal)

        # Eigenenergies should also be the same.
        numpy.testing.assert_almost_equal(
            msmd_fci.eigenenergies, msmd_cas.eigenenergies, decimal=decimal)

    def test_cas_versus_fci(self):
        """
        Check that matrix densities agree between FCI and CASSCF/CASCI with a full active space.
        """
        for name, mol in tqdm(self.create_test_molecules().items()):
            for nstate in [1,2]:
                with self.subTest(molecule=name, nstate=nstate):
                    self.compare_cas_and_fci(mol, nstate=nstate)

    def test_cas_water_8e6o_versus_fci(self):
        """
        Check that matrix densities agree approximately between FCI and CASSCF/CASCI
        when core orbitals are double occupied.
        """
        # water molecule
        mol = pyscf.gto.M(
            atom = 'O  0 0 0; H 0.75 0.00 0.50; H 0.75 0.00 -0.50',
            basis = 'sto-3g',
            # singlet
            spin = 0)
        # Oxygen 1s orbital is closed, CAS consists of 8 electrons in 6 orbitals.
        self.compare_cas_and_fci(
            mol,
            nstate=2, ncas=6, nelecas=8,
            # Maximum deviation is 2 decimals.
            decimal=2)

    def test_integrals_2e2o(self):
        """
        Check that the matrix densities from a CAS(2e/2o) calculation
        still integrate to the correct number of electrons.
        """
        # water molecule
        mol = pyscf.gto.M(
            atom = 'O  0 0 0; H 0.75 0.00 0.50; H 0.75 0.00 -0.50',
            basis = 'sto-3g',
            # singlet
            spin = 0)
        # CAS(2e,2o)
        msmd = self.create_matrix_density(
            mol, nstate=2, ncas=2, nelecas=2)
        self.check_integrals(msmd)

    def test_raises_error(self):
        """
        Check that an error is raised if more states are requested than what
        is possible in the active space.
        """
        # hydrogen molecule
        mol = pyscf.gto.M(
            atom = 'H 0 0 0; H 0 0 0.74',
            basis = 'sto-3g',
            charge = 0,
            spin = 0)
        # CAS(2e,2o) contains 4 Slater determinants, so 5 states is not possible.
        with self.assertRaises(RuntimeError):
            msmd = self.create_matrix_density(
                mol, nstate=5, ncas=2, nelecas=2, raise_error=True)


class TestMultistateMatrixDensityCASSCF(BaseTestMultistateMatrixDensityCAS, TestMultistateMatrixDensityFCI):
    def create_matrix_density(
            self,
            mol,
            nstate=2, ncas=None, nelecas=None,
            spin_symmetry=True, raise_error=True):
        # call the static method
        return MultistateMatrixDensityCASSCF.create_matrix_density(
            mol,
            nstate=nstate, ncas=ncas, nelecas=nelecas,
            spin_symmetry=spin_symmetry, raise_error=raise_error)


class TestMultistateMatrixDensityCASCI(BaseTestMultistateMatrixDensityCAS, TestMultistateMatrixDensityFCI):
    def create_matrix_density(
            self,
            mol,
            nstate=2, ncas=None, nelecas=None,
            spin_symmetry=True, raise_error=True):
        # call the static method
        return MultistateMatrixDensityCASCI.create_matrix_density(
            mol,
            nstate=nstate, ncas=ncas, nelecas=nelecas,
            spin_symmetry=spin_symmetry, raise_error=raise_error)


class TestMultistateMatrixDensityTDDFT(BaseTestMultistateMatrixDensity, unittest.TestCase):
    def create_test_molecules(self):
        """ dictionary with different molecules to run the tests on """
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
            # many electrons
            'water': pyscf.gto.M(
                atom = 'O  0 0 0; H 0.75 0.00 0.50; H 0.75 0.00 -0.50',
                basis = 'sto-3g',
                # singlet
                spin = 0),
            # effective core potential which removes the 1s orbital of oxygen
            'water (ECP)': pyscf.gto.M(
                atom = 'O  0 0 0; H 0.75 0.00 0.50; H 0.75 0.00 -0.50',
                basis = {'O': 'crenbl', 'H': 'sto-3g'},
                ecp = {'O': 'crenbl'},
                # singlet
                spin = 0),
        }
        return molecules

    def create_matrix_density(self, mol, nstate=4):
        # call the static method
        return MultistateMatrixDensityTDDFT.create_matrix_density(mol, nstate=nstate)

    def check_cis_coefficients_normalization(self, mol):
        """
        check that the CIS coefficients are orthonormalized.

        :param mol: A test molecule
        :type mol: gto.Mole
        """
        # Example density.
        msmd = self.create_matrix_density(mol)

        # Overlap matrix between CIS states.
        overlap = numpy.einsum(
            'iov,jov->ij',
            msmd.cis_coefficients,
            msmd.cis_coefficients)
        # The overlap matrix should be the identity matrix.
        numpy.testing.assert_almost_equal(
            overlap, numpy.eye(msmd.number_of_states-1), decimal=8)

    def test_cis_coefficients_normalization(self):
        """ Check CIS coefficients for all test molecules """
        for name, mol in tqdm(self.create_test_molecules().items()):
            with self.subTest(molecule=name):
                self.check_cis_coefficients_normalization(mol)

    def check_transition_dipoles(self, mol):
        """
        Check that the transition densities between the ground state and the
        excited states are correct by comparing the transition dipoles
        from numerical integration with the analytical ones.

          td₀ᵢ = <Ψ₀|r|Ψᵢ> = ∫ r D₀ᵢ(r) dr
        """
        # Example density.
        msmd = self.create_matrix_density(mol)

        # integration grid
        grids = pyscf.dft.gen_grid.Grids(msmd.mol)
        grids.level = 8
        grids.build()

        # evaluate (transition) density matrices
        D, _, _ = msmd.evaluate(grids.coords)
        # integrate transition dipoles between ground state and excited states
        transition_dipoles = numpy.einsum(
            'r,rd,sijr->ijd', grids.weights, grids.coords, D)[0,1:,:]

        # reference transition dipoles
        transition_dipoles_ref = msmd.tddft.transition_dipole()

        numpy.testing.assert_almost_equal(
            transition_dipoles, transition_dipoles_ref, decimal=8)

    @unittest.skip("pyscf CIS coefficients, CIS=2*(X+Y), are not orthonormalized")
    def test_transition_dipoles(self):
        """ Compare numerical and analytical transition dipoles """
        for name, mol in tqdm(self.create_test_molecules().items()):
            with self.subTest(molecule=name):
                self.check_transition_dipoles(mol)


if __name__ == "__main__":
    unittest.main(failfast=True, verbosity=2)
