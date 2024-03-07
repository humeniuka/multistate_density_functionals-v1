#!/usr/bin/env python
# coding: utf-8
"""
The kinetic energy of an (anti)symmetric wavefunction with density 𝜌(r)
is bounded from below by the von-Weizsäcker kinetic energy,

    T[𝜌] ≥ 1/8 ∫ |∇𝜌|²/𝜌.

This is theorem 1.1 in [Lieb]. Applying it to each electronic state i in
the subspace separately, a lower bound on the trace of the kinetic energy
in that subspace emerges,

   1/N ∑ᵢ T[D]ᵢᵢ ≥ 1/N ∑ᵢ 1/8 ∫ |∇Dᵢᵢ|²/Dᵢᵢ.        (lower bound 1)

However, using similar tricks as in [Lieb] another lower bound of
1/N tr(T) can be derived,

    1/N ∑ᵢ T[D]ᵢᵢ ≥ 1/N 1/8 ∫ |∑ᵢ∇Dᵢᵢ|² / (∑ⱼDⱼⱼ)   (lower bound 2)

                  = 1/8 ∫ |1/N ∇tr(D)|² / (1/N tr(D))

                  = 1/8 ∫ |∇ρᵥ(r)|²/ρᵥ(r),

which is equal to the von-Weizsäcker energy of the subspace density ρᵥ = 1/N ∑ᵢ Dᵢᵢ(r).

Except for one-electron systems, the second lower bound seems to be higher.

[Lieb] Lieb, Elliott H. "Density functionals for Coulomb systems."
    Inequalities: Selecta of Elliott H. Lieb (2002): 269-303.
"""
from abc import ABC, abstractmethod
import numpy
import pyscf.dft
import pyscf.gto

from msdft.MultistateMatrixDensity import MultistateMatrixDensity


class LowerBoundKinetic(ABC):
    def __init__(self, mol, level=8):
        """
        The abstract base class for functionals that bound the
        average kinetic energy of the subspace from below,

          1/N ∑ᵢ T[D]ᵢᵢ ≥  bound[D]

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

    @abstractmethod
    def lower_bound(
            self,
            msmd : MultistateMatrixDensity,
            coords : numpy.ndarray) -> numpy.ndarray:
        """
        The lower bound on the average kinetic energy density

          1/N ∑ᵢ KED[D]ᵢᵢ(r) ≥ bound[D](r)

        at each grid point r.
        """
        pass

    def __call__(
            self,
            msmd : MultistateMatrixDensity,
            available_memory=1<<30) -> float:
        """
        compute the lower bound of the averaged trace of the kinetic energy operator
        in the subspace of electronic states. The bound is a functional of the matrix
        density D and satisfies,

          1/N ∑ᵢ Tᵢᵢ = 1/N ∑ᵢ <Ψᵢ|-1/2 ∑ₙ∇ₙ²|Ψᵢ>  ≥  bound[D(r)],

        where Dᵢⱼ(r) is the electronic density of the state Ψᵢ, Dᵢᵢ(r) = ρᵢ(r),
        or the transition density between the states Ψᵢ and Ψⱼ, Dᵢⱼ(r).

        :param msmd: The multistate matrix density in the electronic subspace
           for which the kinetic energy functional should be bounded.
        :type msmd: :class:`~.MultistateMatrixDensity`

        :param available_memory: The amount of memory (in bytes) that can be
           allocated for the kinetic energy density. If more memory is needed,
           the KED bound is evaluated in multiple chunks. (1<<30 corresponds to 1Gb)
           Since more memory is needed for intermediate quantities, this limit
           is only a rough estimate.
        :type available_memory: int

        :return lower_bound: The lower bound on the averaged trace of the
           kinetic energy matrix in the subspace of the electronic states i=1,...,N
        :rtype lower_bound: float
        """
        # number of grid points
        ncoord = self.grids.coords.shape[0]
        # number of electronic states N in the subspace
        nstate = msmd.number_of_states
        # accumulates the lower bound for 1/N ∑ᵢ Tᵢᵢ
        lower_bound = 0.0

        # If the resulting array that holds the bound of the kinetic energy density
        # exceeds `available_memory`, the bound is evaluated on smaller chunks
        # of the grid and summed into the lower bound at the end.
        needed_memory = 50 * 2 * self.grids.coords.itemsize * nstate**2 * ncoord
        number_of_chunks = max(1, (needed_memory + available_memory) // available_memory)
        # There cannot be more chunks than grid points.
        number_of_chunks = min(ncoord, number_of_chunks)

        # Loop over chunks of grid points and associated integration weights.
        for coords, weights in zip(
                numpy.array_split(self.grids.coords, number_of_chunks),
                numpy.array_split(self.grids.weights, number_of_chunks)):

            # Evaluate the bound for 1/N ∑ᵢ KEDᵢᵢ(r) for each grid point.
            lower_bound_r = self.lower_bound(msmd, coords)

            # The lower bound on the 1/N ∑ᵢ Tᵢᵢ is obtained by integrating the bounds
            # on 1/N ∑ᵢ KEDᵢᵢ(r) over all space.
            #
            #   lower_bound = ∫ lower_bound(r) dr
            #
            lower_bound += numpy.einsum('r,r->', weights, lower_bound_r)

        return lower_bound


class LowerBoundKineticSumOverStates(LowerBoundKinetic):
    """
    Lower bound 1.

    The von-Weizsäcker kinetic energy is a lower bound on the kinetic
    energy of a single state. Averaging over all states in the subspace
    gives the bound,

      1/N ∑ᵢ T[D]ᵢᵢ ≥ 1/N ∑ᵢ 1/8 ∫ |∇Dᵢᵢ|²/Dᵢᵢ.
    """
    def lower_bound(
            self,
            msmd : MultistateMatrixDensity,
            coords : numpy.ndarray) -> numpy.ndarray:
        """
        compute the lower bound on the kinetic energy

           lower_bound(r) = 1/N ∑ᵢ 1/8 ∫ |∇Dᵢᵢ(r)|²/Dᵢᵢ(r)

        as the sum of the von-Weizsäcker kinetic energies of the individual states.

        :param msmd: The multistate matrix density in the electronic subspace
           for which the bound on the kinetic energy density should be evaluated.
        :type msmd: :class:`~.MultistateMatrixDensity`

        :param coords: The Cartesian positions at which the bound of the averaged
           kinetic energy density is calculated.
        :type coords: numpy.ndarray of shape (Ncoord,3)

        :return: lower_bound(r), bound on averaged trace of kinetic energy density
        :rtype: numpy.ndarray of shape (Ncoord)
           lower_bound[r] is a lower bound for 1/N ∑ᵢ KEDᵢᵢ(r) at the grid point
           with index r.
        """
        # number of grid points
        ncoord = coords.shape[0]
        # number of electronic states
        nstate = msmd.number_of_states
        # Evaluate D(r) and ∇D(r) on the integration grid.
        D, grad_D, _ = msmd.evaluate(coords)
        # Sum over spins.
        D = numpy.einsum('sijr->ijr', D)
        grad_D = numpy.einsum('sijdr->ijdr', grad_D)

        # Apply the von-Weizsäcker lower bound to each electronic state individually
        # and average over states.
        # 1/N ∑ᵢ 1/8 |∇Dᵢᵢ(r)|²/Dᵢᵢ(r)
        lower_bound = numpy.zeros(ncoord)
        for i in range(0, nstate):
            # average over states
            lower_bound += (1.0/nstate) * (1.0/8.0) * (
                # scalar product of gradients
                numpy.einsum('dr,dr->r', grad_D[i,i,:,:], grad_D[i,i,:,:]) /
                # Avoid division by zero.
                (D[i,i,:] +  1.0e-20)
            )

        return lower_bound


class LowerBoundKineticSubspaceInvariant(LowerBoundKinetic):
    """
    Lower bound 2.

    Lower bound on the average kinetic energy that is invariant under basis transformation
    in the subspace,

       1/N ∑ᵢ T[D]ᵢᵢ  ≥  ∑ᵢ 1/8 ∫ |∇ρᵥ(r)|²/ρᵥ(r)
    """
    def lower_bound(
            self,
            msmd : MultistateMatrixDensity,
            coords : numpy.ndarray) -> numpy.ndarray:
        """
        compute the lower bound on the kinetic energy

           lower_bound(r) = ∑ᵢ 1/8 ∫ |∇ρᵥ(r)|²/ρᵥ(r)

        as the von-Weizsäcker kinetic energy of the subspace density

           ρᵥ = 1/N ∑ᵢ Dᵢᵢ(r).

        :param msmd: The multistate matrix density in the electronic subspace
           for which the bound on the kinetic energy density should be evaluated.
        :type msmd: :class:`~.MultistateMatrixDensity`

        :param coords: The Cartesian positions at which the bounds on the
           kinetic energy density is calculated.
        :type coords: numpy.ndarray of shape (Ncoord,3)

        :return: lower_bound(r), bound on averaged trace of kinetic energy density
        :rtype: numpy.ndarray of shape (Ncoord)
           lower_bound[r] is a lower bound for 1/N ∑ᵢ KEDᵢᵢ(r) at the grid point
           with index r.
        """
        # number of electronic states
        nstate = msmd.number_of_states
        # Evaluate D(r) and ∇D(r) on the integration grid.
        D, grad_D, _ = msmd.evaluate(coords)
        # Sum over spins and average over electronic states to obtain the
        # subspace density ρᵥ = 1/N ∑ᵢ Dᵢᵢ(r)
        subspace_density = (1.0/nstate) * numpy.einsum('siir->r', D)
        # gradient of subspace density, ∇ρᵥ = 1/N ∑ᵢ ∇Dᵢᵢ(r)
        grad_subspace_density = (1.0/nstate) * numpy.einsum('siidr->dr', grad_D)

        # lower bound on the average kinetic energy density
        #  1/N ∑ᵢ KEDᵢᵢ(r) ≥ 1/8 |∇ρᵥ|²/ρᵥ
        lower_bound = (1.0/8.0) * (
            # scalar product of subspace gradient
            numpy.einsum('dr,dr->r', grad_subspace_density, grad_subspace_density) /
            # Avoid division by zero.
            (subspace_density +  1.0e-20)
        )
        return lower_bound
