#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
In the mean field approximation there is a spurious interaction of an electron with its own
copy in the total charge density. In reality it should only interact with the remaining n-1
electrons. In the core region and at the tails of the electron density outside of the molecule
the self-interaction error is particularly bad.

Since in most functionals the self-interaction in the Hartree-term is not cancelled properly
by the exchange/correlation term, the self-interaction error (SIE) leads to a increase of the
electron repulsion in DFA relative to the full CI reference. This constant shift can be removed
by simply subtracting the SIE for each orbital from the electron repulsion energies for each
electronic state.

Let ρ₁ₛ be the density of a 1s core orbital. Then the SIE from that orbital is

  SIE(1s) = 2 x [ 1/2 (ρ₁ₛᵅ|ρ₁ₛᵅ) - 2¹ᐟ³ Cₓ ∫ ρ₁ₛᵅ(r)⁴ᐟ³ dr + ∫ ρ₁ₛᵅ(r) εᶜ(ρ₁ₛᵅ) dr ]

(ρ₁ₛᵅ|ρ₁ₛᵅ)=(1sᵅ1sᵅ|1sᵅ1sᵅ) is the electrostatic interaction of the density with itself.
The factor two comes from the double occupancy of the core orbital.
"""
import becke
import collections
import numpy

import pyscf.data
from pyscf.dft import numint
import pyscf.scf

from msdft.ElectronRepulsionOperators import LDACorrelationLikeFunctional
from msdft.ElectronRepulsionOperators import LSDAExchangeLikeFunctional


class CoreSelfInteractionCorrection(object):
    def __init__(self, mol, level=8):
        """
        The self-interaction error of the core orbitals.

        :param mol: The molecule contains the information about the elements
            and the basis set. The self-interaction corrections from all core
            orbitals in the molecule are added.
        :type mol: pyscf.gto.Mole

        :param level: The level (3-8) controls the number of grid points
            in the integration grid.
        :type level: int
        """
        # The SIE of the core orbitals only depends on the elemental composition of the molecule,
        # the geometry is irrelevant.
        self.mol = mol
        # generate a multicenter integration grid
        self.grids = pyscf.dft.gen_grid.Grids(mol)
        self.grids.level = level
        self.grids.build()

    @staticmethod
    def self_interaction_energy_of_core(element : str, basis: str):
        """
        compute the self-interaction energy of the core electrons for a single
        element using the supplied basis set.

        The core orbitals are determined by a Hartree-Fock calculation on the
        isolated atom.

        :param element: The name of the element (e.g. 'C' or 'N')
            for which the self-interaction of the core electrons
            should be calculated.
        :type element: str

        :param basis: The basis set (e.g. 'sto-3g')
        :type basis: str

        :return:
            The self-interaction energy in Hartree for each core orbital.
            The factor 2 from the double occupation is already included.
        :rtype: float numpy.ndarray of shape (ncore,)
        """
        # Build an isolated atom.
        atom = pyscf.gto.M(
            atom = f'{element}  0 0 0',
            basis = basis,
            charge = 0,
            # Singlet for even, doublet for odd number of electrons.
            spin = pyscf.data.elements.charge(element)%2
        )

        # number of core orbitals
        number_of_core_orbitals = pyscf.data.elements.chemcore(atom)

        # The core orbitals should look very similar to the atomic orbitals.
        # However, it is not guaranteed that the 1s orbital is the first atomic orbital
        # in the basis set. Therefore we run an SCF calculation, but we are only interested
        # in the lowest (or lowest few for heavier atoms) "molecular" orbitals,
        # which are equal to the core orbitals.
        rhf = pyscf.scf.RHF(atom)
        # Supress printing of SCF energy
        rhf.verbose = 0
        # compute self-consistent field
        rhf.kernel()

        # generate a multicenter integration grid for pyscf.
        grids = pyscf.dft.gen_grid.Grids(atom)
        grids.level = 8
        grids.build()

        # The multicenter grid for the becke module is defined by the centers ...
        atomlist = [(atom.atom_charges()[0], atom.atom_coords()[0])]
        # ... and the radial and angular grids.
        becke.settings.radial_grid_factor = 3
        becke.settings.lebedev_order = 23

        # SIE in Hartree for each core orbital.
        self_interaction_energies = []

        for core_orbital in range(0, number_of_core_orbitals):
            #
            # [A] Find the electrostatic potential Vᵅ generated by the
            #     a single electron in the core orbital ρᵅ.
            #

            # A function that evaluates the density of the core-orbital ρᵅ
            # on a grid. This functions is the input for the Poisson solver.
            def core_orbital_density_function(x, y, z):
                # The `becke` module and the `msdft` module use different
                # shapes for the coordinates of the grid points. Before
                # passing the grid to `msdft`, the arrays have to be flattened
                # and reshaped as (ncoord,3). Before returning the density,
                # it has to be brought into the same shape as each input coordinate
                # array.
                coords = numpy.vstack(
                    [x.flatten(), y.flatten(), z.flatten()]).transpose()

                # The atomic orbital values.
                ao_value = numint.eval_ao(atom, coords)

                # Contract with the MO coefficients to get the values
                # of the molecular orbitals on the grid.
                # mo_value[:,mo] are the grid values of the molecular orbital with index `mo`.
                mo_value = numpy.einsum('ra,am->rm', ao_value, rhf.mo_coeff)
                # Density of the core orbital (one of the lowest molecule orbitals)
                # A core orbital should be dominated by a single atomic orbital
                assert abs(rhf.mo_coeff[:,core_orbital]).max() > 0.9, (
                    "Check the core orbitals. A core orbital should be dominated by a single AO")

                # The core orbital density ρᵅ(r) on the grid for solving the Poisson equation.
                core_orbital_density = pow(abs(mo_value[:,core_orbital]), 2)

                # Give it the same shape as the input arrays.
                core_orbital_density = numpy.reshape(core_orbital_density, x.shape)

                return core_orbital_density

            # The solution Vᵅ(r) of the Poisson equation
            #  ∇²Vᵅ(r) = -4π ρᵅ(r)
            # is returned as a callable.
            potential_function = becke.poisson(atomlist, core_orbital_density_function)

            #
            # [B] Compute the electrostatic energy of the core density ρᵅ
            #     in the its own electrostatic potential.
            #

            # The atomic orbital values.
            ao_value = numint.eval_ao(atom, grids.coords)
            # Contract with the MO coefficients to get the values
            # of the molecular orbitals on the grid.
            mo_value = numpy.einsum('ra,am->rm', ao_value, rhf.mo_coeff)
            # density of the core orbital (one of the lowest molecular orbitals)
            core_orbital_density = pow(abs(mo_value[:,core_orbital]), 2)

            # The electrostatic potential is evaluated on the same
            # integration grid as the density.
            core_orbital_potential = potential_function(
                # x
                grids.coords[:,0],
                # y
                grids.coords[:,1],
                # z
                grids.coords[:,2])

            # The density in one core spin orbital should integrate to 1.
            core_density_integral = numpy.sum(grids.weights * core_orbital_density)
            assert abs(core_density_integral - 1.0) < 1.0e-5, "Core orbital should be normalized to 1."

            # Hartree-part of self-interaction,
            # J[ρᵅ] = 1/2 (1sᵅ1sᵅ|1sᵅ1sᵅ)
            #       = 1/2 ∫∫' ρᵅ(r) ρᵅ(r') / |r-r'|
            #       = 1/2 ∫ ρᵅ(r) Vᵅ(r)
            self_interaction_energy_J = 0.5 * numpy.sum(
                grids.weights * core_orbital_density * core_orbital_potential)

            # Exchange-part of self-interaction, -K[ρᵅ] = -2¹ᐟ³ Cₓ ∫ ρᵅ(r)⁴ᐟ³ dr
            prefactor = pow(2.0, 1.0/3.0) * LSDAExchangeLikeFunctional.Cx
            self_interaction_energy_X = -prefactor * numpy.sum(
                grids.weights * pow(core_orbital_density, 4.0/3.0))

            # Correlation-part of self-interaction, C[ρᵅ] = ∫ ρᵅ(r) εᶜ(ρᵅ) dr
            # A single orbital is fully spin-polarizated, therefore the ferromagnetic (spin=1)
            # correlation energy should be used. However, since the difference between the two
            # is less than 10%, the paramagnetic function (spin=0) is used to keep things simple.
            self_interaction_energy_C = numpy.sum(
                grids.weights * LDACorrelationLikeFunctional.correlation_energy_density(core_orbital_density, spin=0)
            )

            # Combine contributions from direct and indirect part of Coulomb energy.
            # The factor 2 comes from the fact that the core electron is doubly occupied.
            # In the density functional approximation, the interaction of each electron
            # with itself is not excluded.
            self_interaction_energy = 2 * (
                self_interaction_energy_J +
                self_interaction_energy_X +
                self_interaction_energy_C
            )

            self_interaction_energies.append(self_interaction_energy)

        return numpy.array(self_interaction_energies)

    def total_self_interaction_error(self) -> float:
        """
        The self-interaction energies (SIE) from all core orbitals in the molecule are
        calculated and summed. The basis set attached to the molecule is used.

        :return: Total core self-interaction energy (in Hartree)
        :rtype: float
        """
        elements = [
            self.mol.atom_pure_symbol(atom_index) for atom_index in range(0, self.mol.natm)
        ]
        # Count how often an element occurs in the molecule
        element_counts = collections.Counter(elements)

        total_core_SIE = 0.0
        # Loop over unique elements in the molecule.
        for element, count in element_counts.items():
            # SIE for all core orbitals of one element.
            core_SIEs = CoreSelfInteractionCorrection.self_interaction_energy_of_core(
                element, self.mol.basis)
            # Sum contributions from all atoms of the same element.
            total_core_SIE += count * numpy.sum(core_SIEs)

        return total_core_SIE
