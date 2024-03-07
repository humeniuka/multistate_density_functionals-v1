#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy
import numpy.linalg as la

import pyscf.dft

from msdft.MultistateMatrixDensity import MultistateMatrixDensity

class NuclearPotentialOperator(object):
    def __init__(self, mol, level=8):
        """
        The external potential is the interaction energy between the electrons
        and the nuclei. The matrix elements of the nuclear potential in the basis
        of electronic states Ψᵢ is

        Vᵢⱼ = <Ψᵢ| ∑ₘ∑ₙ(-Zₘ)/|rₙ-Rₘ| |Ψⱼ> .

        :param mol: The molecule defines the integration grid.
        :type mol: pyscf.gto.Mole

        :param level: The level (3-8) controls the number of grid points
           in the integration grid.
        :type level: int
        """
        # generate a multicenter integration grid
        self.grids = pyscf.dft.gen_grid.Grids(mol)
        self.grids.level = level
        self.grids.build()

    def __call__(
            self,
            msmd : MultistateMatrixDensity):
        """
        compute the matrix of the external (nuclear) potential in the subspace
        of electronic states as

          Vᵢⱼ = ∫ V(r) Dᵢⱼ(r)

        where Dᵢⱼ(r) is the electronic density of the state Ψᵢ, Dᵢᵢ(r) = ρᵢ(r),
        or the transition density between the states Ψᵢ and Ψⱼ, Dᵢⱼ(r).

        :param msmd: The multistate matrix density in the electronic subspace
        :type msmd: :class:`~.MultistateMatrixDensity`

        :return potential_matrix: The nuclear potential matrix Vᵢⱼ in the subspace
           of the electronic states i,j=1,...,nstate
        :rtype potential_matrix: numpy.ndarray of shape (nstate,nstate)
        """
        # number of grid points
        ncoord = self.grids.coords.shape[0]
        # number of electronic states in the subspace
        nstate = msmd.number_of_states
        # up or down spin
        nspin = 2

        # Evaluate D(r) on the integration grid.
        D, _, _ = msmd.evaluate(self.grids.coords)

        # Evaluate the nuclear potential on the integration grid.
        V = numpy.zeros_like(self.grids.weights)

        # Loop over nuclei in the molecule.
        for nuclear_charge, nuclear_coords in zip(
                msmd.mol.atom_charges(), msmd.mol.atom_coords()):
            # ∑ₘ(-Zₘ)/|r-Rₘ|
            r_nuc_elec = la.norm(self.grids.coords - nuclear_coords, axis=1)
            V += (-nuclear_charge)/r_nuc_elec

        # matrix element of the external potential
        potential_matrix = numpy.zeros((nstate,nstate))

        # Loop over spins. The potential energy is computed separately for each spin
        # projection and added.
        for s in range(0, nspin):
            if numpy.all(D[s,...] == 0.0):
                # There are no electrons with spin projection s
                # that could contribute to the potential energy.
                continue

            # The matrix of the nuclear attraction energy in the subspace is obtained
            # by integration V(r) D_{i,j}(r) over space:
            #
            #  ∫ V(r) Dᵢⱼ(r)
            #
            potential_matrix += numpy.einsum('r,ijr->ij', self.grids.weights * V, D[s,...])

        return potential_matrix
