#!/usr/bin/env python
# coding: utf-8
"""
Lower bound on the electron-repulsion energy.
"""
from abc import ABC, abstractmethod
import becke
import numpy
import pyscf.dft

from msdft.MultistateMatrixDensity import MultistateMatrixDensity


class LiebOxfordBound(object):
    # The constant cᴸᴼ for the improved lower bound on page 428 of [Lieb&Oxford]
    Lieb_Oxford_constant = 1.68

    def __init__(self, mol, level=8):
        """
        The Lieb-Oxford bound is a lower limit on the expectation value of the
        electron repulsion operator. Suppose that ρ is the electron density of
        a wavefunction Ψ, then the total electron repulsion is bounded from
        below,

          <Ψ|∑ᵦ<ᵧ 1/|rᵦ-rᵧ||Ψ>  ≥  1/2 ∫∫ ρ(r) ρ(r') / |r-r'|  -  cᴸᴼ ∫ ρ(r)⁴ᐟ³,

        where cᴸᴼ = 1.68 is derived in reference [Lieb&Oxford].
        Note that the above definition of the bound contains both the (trivial)
        direct part and the indirect part of the Coulomb energy.

        References
        ----------
        [Lieb&Oxford] Lieb, Elliott H., and Stephen Oxford.
             "Improved lower bound on the indirect Coulomb energy."
             Int. J. Quant. Chem. 19.3 (1981): 427-439.

        :param mol: The molecule defines the integration grid.
        :type mol: pyscf.gto.Mole

        :param level: The level (3-8) controls the number of grid points
           in the integration grid.
        :type level: int
        """
        self.mol = mol
        # generate a multicenter integration grid
        self.grids = pyscf.dft.gen_grid.Grids(mol)
        self.grids.level = level
        self.grids.build()

    def direct_coulomb_energy(
            self,
            density_function: callable) -> float:
        """
        Compute the direct part of the electron repulsion

          J[ρ] = 1/2 ∫∫ ρ(r) ρ(r') / |r-r'|

        by solving the Poisson equation for ρ on a multicenter grid.

        :param density_function: A function that evaluates ρ at grid points.
            `density_function(x,y,z)` should take three arrays `x`, `y` and `z` with the
            Cartesian coordinates as arguments and return the values of ρ(x,y,z) as an
            array of that same shape as `x`.
        :type density_function: callable

        :return hartree_energy: The Hartree (or direct) energy J[ρ]
        :rtype hartree_energy: float
        """
        # Cartesian coordinates of grid points.
        x, y, z = self.grids.coords[:,0], self.grids.coords[:,1], self.grids.coords[:,2]

        # For solving the Poisson equation another multicenter Becke grid
        # is employed, which can have a different number of angular and radial
        # grid points than the one for doing the integrals.

        # The multicenter grid is defined by the centers ...
        atomlist = []
        for nuclear_charge, nuclear_coords in zip(
                self.mol.atom_charges(), self.mol.atom_coords()):
            atomlist.append((int(nuclear_charge), nuclear_coords))
        # ... and the radial and angular grids.
        becke.settings.radial_grid_factor = 3
        becke.settings.lebedev_order = 23

        # [1] The Poisson equation is solved for ρ.
        # The solution of the Poisson equation
        #  ∇²V(r) = -4π ρ(r)
        # is returned as a callable.
        potential_function = becke.poisson(atomlist, density_function)

        # The electrostatic potential is evaluated on the same
        # integration grid as the density.
        V = potential_function(x, y, z)

        # Evaluate ρ(r) on the integration grid.
        density = density_function(x, y, z)
        # [2] Integrate eletrostatic energy.
        #
        #  J[ρ(r)] = 1/2 ∫ ρ(r) V(r)
        #
        hartree_energy = 0.5 * numpy.sum(self.grids.weights * density * V)

        return hartree_energy

    def bound_on_indirect_energy(
            self,
            density_function: callable) -> float:
        """
        The Lieb-Oxford bound on the indirect part of the Coulomb energy,

          E[ρ] ≥  - cᴸᴼ ∫ ρ(r)⁴ᐟ³

        with cᴸᴼ = 1.68.

        :param density_function: A function that evaluates ρ at grid points.
            `density_function(x,y,z)` should take three arrays `x`, `y` and `z` with the
            Cartesian coordinates as arguments and return the values of ρ(x,y,z) as an
            array of that same shape as `x`.
        :type density_function: callable

        :return lower_bound: Lower bound on indirect energy E[ρ]
        :rtype lower_bound: float
        """
        # Cartesian coordinates of grid points.
        x, y, z = self.grids.coords[:,0], self.grids.coords[:,1], self.grids.coords[:,2]
        # Evaluate ρ(r) on the integration grid.
        density = density_function(x, y, z)

        lower_bound = -self.Lieb_Oxford_constant * \
            numpy.sum(self.grids.weights * pow(density, 4.0/3.0))

        return lower_bound

    def __call__(
            self,
            density_function: callable) -> float:
        """
        Evaluate the Lieb-Oxford bound for a density ρ

           <Ψ|∑ᵦ<ᵧ 1/|rᵦ-rᵧ||Ψ>  ≥  1/2 ∫∫ ρ(r) ρ(r') / |r-r'|  -  cᴸᴼ ∫ ρ(r)⁴ᐟ³

        :param density_function: A function that evaluates ρ at grid points.
            `density_function(x,y,z)` should take three arrays `x`, `y` and `z` with the
            Cartesian coordinates as arguments and return the values of ρ(x,y,z) as an
            array of that same shape as `x`.
        :type density_function: callable

        :return lower_bound: Lower bound on electron repulsion energy
        :rtype lower_bound: float
        """
        # direct part of Coulomb energy
        J = self.direct_coulomb_energy(density_function)
        # lower bound on indirect part of Coulomb energy
        lower_bound_indirect = self.bound_on_indirect_energy(density_function)

        lower_bound = J + lower_bound_indirect

        return lower_bound


class LowerBoundElectronRepulsion(ABC):
    def __init__(self, mol, level=8):
        """
        The abstract base class for functionals that bound the
        average electron repulsion energy of the subspace from below,

          1/N ∑ᵢ <Ψᵢ|∑ᵦ<ᵧ 1/|rᵦ-rᵧ||Ψᵢ> ≥ bound[D]

        :param mol: The molecule defines the integration grid.
        :type mol: pyscf.gto.Mole

        :param level: The level (3-8) controls the number of grid points
           in the integration grid.
        :type level: int
        """
        self.lieb_oxford_bound = LiebOxfordBound(mol, level=level)

    @abstractmethod
    def __call__(
            self,
            msmd : MultistateMatrixDensity) -> float:
        """
        Compute the lower bound as a function of the mulistate
        matrix density.
        """
        pass


class LowerBoundElectronRepulsionSumOverStates(LowerBoundElectronRepulsion):
    """
    The Lieb-Oxford bound is applied to each electronic state separately
    and then the different bounds are averaged over all states.

        1/N ∑ᵢ <Ψᵢ|∑ᵦ<ᵧ 1/|rᵦ-rᵧ||Ψᵢ>  ≥  1/N ∑ᵢ ( J[Dᵢᵢ] - cᴸᴼ ∫ Dᵢᵢ(r)⁴ᐟ³ )
    """
    def __call__(
            self,
            msmd : MultistateMatrixDensity) -> float:
        """
        compute the lower bound on the electron repulsion by averaging over
        Lieb-Oxford bounds for individual states,

          1/N ∑ᵢ ( J[Dᵢᵢ] - cᴸᴼ ∫ Dᵢᵢ(r)⁴ᐟ³ )

        :param msmd: The multistate matrix density in the electronic subspace
           for which the bound on the electron repulsion should be evaluated.
        :type msmd: :class:`~.MultistateMatrixDensity`

        :return: The lower bound on direct and indirect Coulomb energy
        :rtype: float
        """
        # number of electronic states
        nstate = msmd.number_of_states
        # Compute the Lieb-Oxford bound for each state
        lower_bound = 0.0
        for i in range(0, nstate):
            # Density Dᵢᵢ(r) of state i,
            def density_function_i(x, y, z):
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
                density = spin_density[0,i,i,:] + spin_density[1,i,i,:]

                # Give it the same shape as the input arrays.
                density = numpy.reshape(density, x.shape)
                # Dᵢᵢ(r)
                return density

            # Average over states
            lower_bound += (1.0/nstate) * self.lieb_oxford_bound(density_function_i)

        return lower_bound


class LowerBoundElectronRepulsionSubspaceInvariant(LowerBoundElectronRepulsion):
    """
    The Lieb-Oxford bound is applied to the subspace density,

       ρᵥ(r) = 1/N ∑ᵢ Dᵢᵢ(r)

    so that the lower bound on the average electron repulsion energy becomes

        1/N ∑ᵢ <Ψᵢ|∑ᵦ<ᵧ 1/|rᵦ-rᵧ||Ψᵢ>  ≥  J[ρᵥ] - cᴸᴼ ∫ ρᵥ(r)⁴ᐟ³.
    """
    def __call__(
            self,
            msmd : MultistateMatrixDensity) -> float:
        """
        compute the lower bound on the electron repulsion from the subspace density

          J[ρᵥ] - cᴸᴼ ∫ ρᵥ(r)⁴ᐟ³

        :param msmd: The multistate matrix density in the electronic subspace
           for which the bound on the electron repulsion should be evaluated.
        :type msmd: :class:`~.MultistateMatrixDensity`

        :return: The lower bound on direct and indirect Coulomb energy
        :rtype: float
        """
        # number of electronic states
        nstate = msmd.number_of_states
        # Function for evaluating the subspace density ρᵥ(r) = 1/N ∑ᵢ Dᵢᵢ(r)
        def subspace_density_function(x, y, z):
            coords = numpy.vstack(
                [x.flatten(), y.flatten(), z.flatten()]).transpose()

            # Evaluate the density.
            spin_density, _, _ = msmd.evaluate(coords)
            # number of grid points
            ncoord = spin_density.shape[-1]

            # Average state densities to get ρᵥ(r).
            subspace_density = numpy.zeros(ncoord)
            # Loop over states
            for i in range(0, nstate):
                # The Coulomb potential does not distinguish spins, so
                # sum over spins.
                # ρᵥ(r) = 1/N ∑ᵢ Dᵢᵢ(r)
                subspace_density += 1.0/nstate * (
                    spin_density[0,i,i,:] + spin_density[1,i,i,:])

            # Give it the same shape as the input arrays.
            subspace_density = numpy.reshape(subspace_density, x.shape)

            return subspace_density

        # Lieb-Oxford bound for subspace density
        lower_bound = self.lieb_oxford_bound(subspace_density_function)

        return lower_bound
