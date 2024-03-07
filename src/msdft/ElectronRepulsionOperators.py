#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Matrix elements of the electron-repulsion operator between two
n-electron wavefunctions Ψᵢ and Ψⱼ

    Wᵢⱼ = ∫dx1 ∫dx2...∫dxn Ψ*ᵢ(x1,x2,...,xn) ∑ᵦ<ᵧ 1/|rᵦ-rᵧ| Ψⱼ(x1,x2,...,xn)

with xᵦ=(rᵦ,σᵦ), are approximated a matrix functional of the matrix density,
i.e. W[D(r)].
"""
from abc import ABC, abstractmethod

try:
    import becke
except ImportError as err:
    print("""
    Solution of the Poisson equation requires the `becke` module
    which can be obtained from
       `https://github.com/humeniuka/becke_multicenter_integration`
    or can be installed from PyPI via
       pip install becke-multicenter-integration
    """)
    raise err

import numpy
import numpy.linalg as la
import pyscf.dft

from msdft.MultistateMatrixDensity import MultistateMatrixDensity


class HartreeLikeFunctional(object):
    def __init__(self, mol):
        """
        The Hartree energy is the interaction of a classical charge density with itself.
        There is no equivalent of the Hartree term in the multistate density functional
        formalism, because the Hartree part of the Coulomb energy does not correspond to
        an operator. We can define a functional of the matrix density D(r) that reduces
        to the Hartree energy in the case of a single electronic state, which is an
        analytical functional and transforms correctly under basis transformations U,
        i.e. `U⁻¹ J[D(r)] U = J[U⁻¹ D(r) U]` :

          J[D(r)]ᵢⱼ = 1/2 ∑ₖ ∫∫' Dᵢₖ(r) Dₖⱼ(r')/|r-r'|

        If there are multiple electronic state, the diagonals J[D(r)]ᵢᵢ differ from the
        classical Coulomb interaction of the state density of state i with itself, because
        the matrix product ∑ₖDᵢₖ(r) Dₖⱼ(r') mixes in transition densities to other
        states k≠i.

        :param mol: Not used.
        :type mol: pyscf.gto.Mole
        """
        pass

    def __call__(
            self,
            msmd : MultistateMatrixDensity):
        """
        compute the Hartree-like part of the electron-electron repulsion operator
        in the subspace of electronic states,

          J[D(r)]ᵢⱼ = 1/2 ∑ₖ ∫∫' Dᵢₖ(r) Dₖⱼ(r')/|r-r'|

        where Dᵢⱼ(r) is the electronic density of the state Ψᵢ, Dᵢᵢ(r) = ρᵢ(r),
        or the transition density between the states Ψᵢ and Ψⱼ, Dᵢⱼ(r).
        """
        return msmd.hartree_matrix_product()


class HartreeLikeFunctionalPoisson(object):
    def __init__(self, mol, level=8):
        """
        The Hartree energy is the interaction of a classical charge density with itself.
        There is no equivalent of the Hartree term in the multistate density functional
        formalism, because the Hartree part of the Coulomb energy does not correspond to
        an operator. We can define a functional of the matrix density D(r) that reduces
        to the Hartree energy in the case of a single electronic state, which is an
        analytical functional and transforms correctly under basis transformations U,
        i.e. `U⁻¹ J[D(r)] U = J[U⁻¹ D(r) U]` :

          J[D(r)]ᵢⱼ = 1/2 ∑ₖ ∫∫' Dᵢₖ(r) Dₖⱼ(r')/|r-r'|

        If there are multiple electronic state, the diagonals J[D(r)]ᵢᵢ differ from the
        classical Coulomb interaction of the state density of state i with itself, because
        the matrix product ∑ₖDᵢₖ(r) Dₖⱼ(r') mixes in transition densities to other
        states k≠i.

        NOTE: :class:`~.HartreeLikeFunctionalPoisson` is much slower than :class:`~.HartreeLikeFunctional`
            because the Poisson equation is solved numerically on a grid.

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
        compute the Hartree-like part of the electron-electron repulsion operator
        in the subspace of electronic states,

          J[D(r)]ᵢⱼ = 1/2 ∑ₖ ∫∫' Dᵢₖ(r) Dₖⱼ(r')/|r-r'|

        where Dᵢⱼ(r) is the electronic density of the state Ψᵢ, Dᵢᵢ(r) = ρᵢ(r),
        or the transition density between the states Ψᵢ and Ψⱼ, Dᵢⱼ(r).

        The 6-dimensional integral is calculated by solving the Poisson equation
        for the electrostatic potential generated by one of the (transition) densities,

          J[D(r)]ᵢⱼ = 1/2 ∑ₖ ∫ Dᵢₖ(r) Vₖⱼ(r)

        where

          ∇²Vₖⱼ(r) = -4π Dₖⱼ(r)

        is solved on a multicenter Becke grid.

        :param msmd: The multistate matrix density in the electronic subspace
        :type msmd: :class:`~.MultistateMatrixDensity`

        :return hartree_like_matrix: The Hartree-like matrix Jᵢⱼ in the subspace
           of the electronic states i,j=1,...,nstate
        :rtype hartree_like_matrix: numpy.ndarray of shape (nstate,nstate)
        """
        # number of grid points
        ncoord = self.grids.coords.shape[0]
        # number of electronic states in the subspace
        nstate = msmd.number_of_states

        # Evaluate D(r) on the integration grid.
        D, _, _ = msmd.evaluate(self.grids.coords)

        # For solving the Poisson equation another multicenter Becke grid
        # is employed, which can have a different number of angular and radial
        # grid points than the one for doing the integrals.

        # The multicenter grid is defined by the centers ...
        atomlist = []
        for nuclear_charge, nuclear_coords in zip(
                msmd.mol.atom_charges(), msmd.mol.atom_coords()):
            atomlist.append((int(nuclear_charge), nuclear_coords))
        # ... and the radial and angular grids.
        becke.settings.radial_grid_factor = 3
        becke.settings.lebedev_order = 23

        # [1] The Poisson equation is solved for each Dₖⱼ(r).
        V = numpy.zeros((nstate,nstate,ncoord))
        for k in range(0, nstate):
            for j in range(0, nstate):
                # A function for evaluating the (transition) density Dₖⱼ(r) on
                # the multicenter Becke grid.
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
                    density = spin_density[0,k,j,:]+ spin_density[1,k,j,:]

                    # Give it the same shape as the input arrays.
                    density = numpy.reshape(density, x.shape)
                    return density

                # The solution of the Poisson equation
                #  ∇²potential(r) = -4π density(r)
                # is returned as a callable.
                potential_function = becke.poisson(atomlist, density_function)

                # The electrostatic potential is evaluated on the same
                # integration grid as the density.
                V[k,j,:] = potential_function(
                    # x
                    self.grids.coords[:,0],
                    # y
                    self.grids.coords[:,1],
                    # z
                    self.grids.coords[:,2])

        # [2] Integrate and do the matrix-matrix product.
        #
        #  J[D(r)]ᵢⱼ = 1/2 ∑ₖ ∫ Dᵢₖ(r) Vₖⱼ(r)
        #
        hartree_like_matrix = 0.5 * numpy.einsum(
            'r,ikr,kjr->ij',
            self.grids.weights,
            # Both spins feel the same Coulomb potentials, so sum over spins.
            D[0,...]+D[1,...],
            V)

        return hartree_like_matrix


class ExchangeCorrelationLikeFunctional(ABC):
    def __init__(self, mol, level=8):
        """
        The abstract base class for multi-state exchange/correlation functionals.

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
    def energy_density(
            self,
            msmd : MultistateMatrixDensity,
            coords : numpy.ndarray):
        pass

    def __call__(
            self,
            msmd : MultistateMatrixDensity,
            available_memory=1<<30):
        """
        compute the exchange/correlation-like contribution to the electron repulsion operator
        in the subspace of electronic states by evaluating the exchange/correlation energy
        functional XC[D(r)] on the matrix density D(r):

          XCᵢⱼ = XC[D(r)]ᵢⱼ = ∫ xc[D]ᵢⱼ(r) dr,

        where Dᵢⱼ(r) is the electronic density of the state Ψᵢ, Dᵢᵢ(r) = ρᵢ(r),
        or the transition density between the states Ψᵢ and Ψⱼ, Dᵢⱼ(r).
        xc[D](r) is a local approximation of the exchange/correlation energy density (XCED).

        :param msmd: The multistate matrix density in the electronic subspace
           for which the exchange energy functional should be evaluated.
        :type msmd: :class:`~.MultistateMatrixDensity`

        :param available_memory: The amount of memory (in bytes) that can be
           allocated for the exchange/correlation energy density. If more memory is needed,
           the XED is evaluated in multiple chunks. (1<<30 corresponds to 1Gb)
           Since more memory is needed for intermediate quantities, this limit
           is only a rough estimate.
        :type available_memory: int

        :return xc_like_matrix: The exchange/correlation energy matrix XCᵢⱼ in the subspace
           of the electronic states i,j=1,...,nstate
        :rtype xc_like_matrix: numpy.ndarray of shape (nstate,nstate)
        """
        # number of grid points
        ncoord = self.grids.coords.shape[0]
        # number of electronic states in the subspace
        nstate = msmd.number_of_states
        # matrix element of the exchange/correlation-like part of the electron repulsion operator.
        xc_like_matrix = numpy.zeros((nstate,nstate))

        # If the resulting array that holds the exchange/correlation energy density
        # exceeds `available_memory`, the XCED is evaluated on smaller chunks
        # of the grid and summed into the kinetic matrix at the end.
        needed_memory = 50 * 2 * xc_like_matrix.itemsize * nstate**2 * ncoord
        number_of_chunks = max(1, (needed_memory + available_memory) // available_memory)
        # There cannot be more chunks than grid points.
        number_of_chunks = min(ncoord, number_of_chunks)

        # Loop over chunks of grid points and associated integration weights.
        for coords, weights in zip(
                numpy.array_split(self.grids.coords, number_of_chunks),
                numpy.array_split(self.grids.weights, number_of_chunks)):

            # Evaluate the exchange/correlation energy density on the grid.
            XCED = self.energy_density(msmd, coords)

            # The matrix of the exchange/correlation-like part of the electron-repulsion
            # operator in the subspace is obtained by integration of XEDᵢⱼ(r) over space and spin
            #
            #   XCᵢⱼ = ∫ XCEDᵢⱼ(r) dr
            #
            xc_like_matrix += numpy.einsum('r,sijr->ij', weights, XCED)

        return xc_like_matrix


# Cₓ = (3/4) (3/pi)¹ᐟ³ = 0.738 from Dirac's exchange-energy, Eqn. (6.1.20) in [Parr&Yang]
Cx_Dirac = 0.7386
# Cₓ from the "Gaussian" approximation in Eqn. (6.5.25) of [Parr&Yang]
Cx_Gaussian = 0.7937

class LSDAExchangeLikeFunctional(ExchangeCorrelationLikeFunctional):
    # The prefactor Cₓ for the exchange energy.
    Cx = Cx_Dirac

    def __init__(self, mol, level=8):
        """
        Multi-state exchange energy according to the local-spin density approximation
        (eqn. 8.2.16 in Ref. [Yang&Parr]),

        K[D(r)] = 2¹ᐟ³ Cₓ ∫ [ Dᵅ(r)⁴ᐟ³ + Dᵝ(r)⁴ᐟ³ ] dr

        D(r)⁴ᐟ³ is a fractional matrix-power of D(r), which is calculated by diagonalizing D.

        The value of the prefactor Cₓ = 0.7386 is taken from Dirac's approximation in
        Eqn. (6.1.20) of chapter 6 in Ref. [Yang&Parr]

        References
        ----------
        [Yang&Parr] Parr & Yang (1989), "Density Functional Theory of Atoms and Molecules".

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

    def energy_density(
            self,
            msmd : MultistateMatrixDensity,
            coords : numpy.ndarray):
        """
        compute the energy density for the exchange-like part of the electron-electron
        repulsion operator in the subspace of electronic states,

          XED[D]ᵢⱼ(r) = 2¹ᐟ³ Cₓ [D(r)⁴ᐟ³]ᵢⱼ

        NOTE: At odds with the usual definition of the exchange energy density,
        (εₓ,ᵢⱼ(r) ∝ ρ(r)¹ᐟ³), XED contains an additional factor of D(r)
        (XED(r) ∝ D(r)⁴ᐟ³), since the exchange energy is calculated
        as K[D] = ∫ XED(r) dr rather than K[ρ] = ∫ ρ(r) εₓ(r) dr.

        :param msmd: The multistate matrix density in the electronic subspace
        :type msmd: :class:`~.MultistateMatrixDensity`

        :param coords: The Cartesian positions at which the exchange energy
           density is calculated.
        :type coords: numpy.ndarray of shape (Ncoord,3)

        :return: XEDᵢⱼ(r), exchange energy density
        :rtype: numpy.ndarray of shape (2,Mstate,Mstate,Ncoord)
           XED[s,i,j,r] is the exchange energy density with spin s,
           between the electronic states i and j at position coords[r,:].
        """
        # number of grid points
        ncoord = coords.shape[0]
        # number of electronic states in the subspace
        nstate = msmd.number_of_states
        # up or down spin
        nspin = 2

        # exchange-energy density  XED[Dᵅ]ᵢⱼ(r) = 2¹ᐟ³ Cₓ [Dᵅ(r)⁴ᐟ³]ᵢⱼ
        XED = numpy.zeros((nspin,nstate,nstate,ncoord))

        # Evaluate D(r) on the integration grid.
        D, _, _ = msmd.evaluate(coords)

        # Trace over electronic states to get tr(D)(r).
        # `trace_D` has shape (2,Ncoord,), trace_D[s,:] = sum_i D[spin,i,i,:]
        trace_D = numpy.einsum('siir->sr', D)

        # Loop over spins. The exchange energy is computed separately for each spin
        # projection and added.
        for s in range(0, nspin):
            if numpy.all(trace_D[s,...] == 0.0):
                # There are no electrons with spin projection s
                # that could contribute to the exchange energy.
                continue
            prefactor = pow(2.0, 1.0/3.0) * self.Cx
            for r in range(0, ncoord):
                # Compute eigenvalues Λ and eigenvectors U of the symmetric matrix D.
                L, U = numpy.linalg.eigh(D[s,:,:,r])
                # Numerical rounding errors might produce tiny, negative eigenvalues instead of 0.
                assert numpy.all(L > -1.0e-12), "Eigenvalues of matrix density D are expected to be positive."

                # The fractional matrix power is obtained from the eigenvalue decomposition
                # as D⁴ᐟ³(r) = U(r) Λ⁴ᐟ³(r) Uᵀ(r)
                D_matrix_power = numpy.einsum('ia,a,ja->ij', U, pow(abs(L), 4.0/3.0), U)
                # LSDA exchange energy density
                exchange_energy_r = prefactor * D_matrix_power
                # Check that the exchange energy density is real.
                assert numpy.sum(abs(exchange_energy_r.imag)) < 1.0e-10

                XED[s,:,:,r] = exchange_energy_r

        return XED

class LDAExchangeLikeFunctional(ExchangeCorrelationLikeFunctional):
    # The prefactor Cₓ for the exchange energy.
    Cx = Cx_Dirac

    def __init__(self, mol, level=8):
        """
        Multi-state exchange energy according to the local density approximation
        (eqn. 6.5.29 in Ref. [Yang&Parr]),

        K[D(r)] = Cₓ ∫ D(r)⁴ᐟ³ dr

        D(r)⁴ᐟ³ is a fractional matrix-power of D(r), which is calculated by diagonalizing D.

        The value of the prefactor Cₓ = 0.7386 is taken from Dirac's approximation in
        Eqn. (6.1.20) of chapter 6 in Ref. [Yang&Parr]

        References
        ----------
        [Yang&Parr] Parr & Yang (1989), "Density Functional Theory of Atoms and Molecules".

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

    def energy_density(
            self,
            msmd : MultistateMatrixDensity,
            coords : numpy.ndarray):
        """
        compute the energy density for the exchange-like part of the electron-electron
        repulsion operator in the subspace of electronic states,

          XED[D]ᵢⱼ(r) = Cₓ [D(r)⁴ᐟ³]ᵢⱼ

        NOTE: At odds with the usual definition of the exchange energy density,
        (εₓ,ᵢⱼ(r) ∝ ρ(r)¹ᐟ³), XED contains an additional factor of D(r)
        (XED(r) ∝ D(r)⁴ᐟ³), since the exchange energy is calculated
        as K[D] = ∫ XED(r) dr rather than K[ρ] = ∫ ρ(r) εₓ(r) dr.

        :param msmd: The multistate matrix density in the electronic subspace
        :type msmd: :class:`~.MultistateMatrixDensity`

        :param coords: The Cartesian positions at which the exchange energy
           density is calculated.
        :type coords: numpy.ndarray of shape (Ncoord,3)

        :return: XEDᵢⱼ(r), exchange energy density
        :rtype: numpy.ndarray of shape (1,Mstate,Mstate,Ncoord)
           XED[0,i,j,r] is the exchange energy density between the
           electronic states i and j at position coords[r,:].
           There is only one spin component, since XED operates on the spin-traced
           density.
        """
        # number of grid points
        ncoord = coords.shape[0]
        # number of electronic states in the subspace
        nstate = msmd.number_of_states

        # exchange-energy density  XED[D]ᵢⱼ(r) = Cₓ [D(r)⁴ᐟ³]ᵢⱼ
        XED = numpy.zeros((1,nstate,nstate,ncoord))

        # Evaluate Dᵅ(r) on the integration grid.
        D, _, _ = msmd.evaluate(coords)

        # Sum over spins to get total density D = Dᵅ(r) + Dᵝ(r).
        total_density = D[0,...] + D[1,...]
        for r in range(0, ncoord):
            # Compute eigenvalues Λ and eigenvectors U of the symmetric matrix D.
            L, U = numpy.linalg.eigh(total_density[:,:,r])
            # Numerical rounding errors might produce tiny, negative eigenvalues instead of 0.
            assert numpy.all(L > -1.0e-12), "Eigenvalues of matrix density D are expected to be positive."

            # The fractional matrix power is obtained from the eigenvalue decomposition
            # as D⁴ᐟ³(r) = U(r) Λ⁴ᐟ³(r) Uᵀ(r)
            D_matrix_power = numpy.einsum('ia,a,ja->ij', U, pow(abs(L), 4.0/3.0), U)
            # LDA exchange energy density
            exchange_energy_r = self.Cx * D_matrix_power
            # Check that the exchange energy density is real.
            assert numpy.sum(abs(exchange_energy_r.imag)) < 1.0e-10

            XED[0,:,:,r] = exchange_energy_r

        return XED


class LDACorrelationLikeFunctional(ExchangeCorrelationLikeFunctional):
    # Parameters of Chachiyo's functional from Eqn.(3) of [Chachiyo]
    a = (numpy.log(2.0)-1.0)/(2*numpy.pi**2)
    # b from Eqn.(3) for the paramagnetic part εᶜ₀
    b_paramagnetic = 20.4562557
    # b from Eqn.(12) for the ferromagnetic part εᶜ₁
    b_ferromagnetic = 27.4203609

    def __init__(self, mol, level=8):
        """
        Multi-state correlation energy according to the local density approximation.

        For a single electronic state it reduces to the correlation energy of the uniform
        electron gas. The functional form from [Chachiyo] is a simple and elegant parameterization
        of the correlation energy per electron of the uniform electron gas.
        It recovers the exact high density limit and fits the quantum Monte-Carlo results of
        [Ceperley&Alder] in the medium density range rather well.

        Taking the paramagnetic part of the correlation energy (spin polarization = 0) and
        replacing the electron density ρ(r) with the density matrix D(r), the multistate extension
        of the Chachiyo functional (Eqn.8 in [Chachiyo]) can be written in the following form:

          C[D(r)] = a ∫ log(Id + b₁ D(r)¹ᐟ³ + b₂ D(r)²ᐟ³ ) D(r) dr

        with

          a = (log(2)-1)/(2 π²) = -0.01554534543482745
          b = 20.4562557 (paramagnetic)
          b₁ = (4π/3)¹ᐟ³ b = 32.975319597703546
          b₂ = (4π/3)²ᐟ³ b = 53.155949872619715

        `f[D] = log(Id + b₁ D(r)¹ᐟ³ + b₂ D(r)²ᐟ³) D(r)` is a matrix funtional,
        which is calculated by diagonalizing D and applying the function f to the eigenvalues.

        References
        ----------
        [Chachiyo] T. Chachiyo (2016), J. Chem. Phys. 145, 2
            "Communication: Simple and accurate uniform electron gas correlation energy for the full range of densities"
        [Ceperley&Alder] D. Ceperley, B. Alder (1980), Phys. Rev. Lett., 45, 7, 566.
            "Ground state of the electron gas by a stochastic method"

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

    @classmethod
    def correlation_energy_density(
            cls,
            density : numpy.ndarray,
            spin=0) ->  numpy.ndarray:
        """
        The correlation energy density of a uniform electron gas in Chachiyo's parameterization.
        In terms of the Wigner-Seitz radius
          rₛ = (4π/3 ρ)⁻¹ᐟ³
        the correlation energy per electron is expressed as
          εᶜ(rₛ) = a log(1+b/rₛ+b/rₛ²)
        The correlation energy density is then given by
          ced(ρ) = ρ(r) εᶜ

        :param density: The total electron density ρ = ρᵅ+ρᵝ
          at the grid points
        :type density: numpy.ndarray of arbitrary shape
        
        :param spin: The spin parameter determines whether the paramagnetic (spin=0)
          or ferromagnetic (spin=1) correlation energy is calculated.
        :type spin: int 

        :return: Electron correlation energy CED
        :rtype: numpy.ndarray of same shape as `density`.
        """
        assert spin in [0,1]
        # The parameter b is different from paramagnetic or ferromagnetic densities.
        if spin == 1:
            b = cls.b_ferromagnetic
        else:
            b = cls.b_paramagnetic

        b1 = pow(4.0/3.0*numpy.pi, 1.0/3.0) * b
        b2 = pow(4.0/3.0*numpy.pi, 2.0/3.0) * b
        # In terms of the density the correlation energy becomes
        #  εᶜ(ρ) = a log( 1 + b1 ρ¹ᐟ³ + b2 ρ²ᐟ³ )
        epsilon_c = cls.a * numpy.log(1.0 + b1 * pow(density, 1.0/3.0) + b2 * pow(density, 2.0/3.0))
        # Multiply the correlation energy per particle by the particel density
        # to get the correlation energy density (CED(r))
        ced = epsilon_c * density
        return ced

    def energy_density(
            self,
            msmd : MultistateMatrixDensity,
            coords : numpy.ndarray):
        """
        compute the energy density for the correlation-like part of the electron-electron
        repulsion operator in the subspace of electronic states,

          CED[D]ᵢⱼ(r) = a log(Id + b₁ D(r)¹ᐟ³ + b₂ D(r)²ᐟ³) D(r)

        :param msmd: The multistate matrix density in the electronic subspace
        :type msmd: :class:`~.MultistateMatrixDensity`

        :param coords: The Cartesian positions at which the correlation energy
           density is calculated.
        :type coords: numpy.ndarray of shape (Ncoord,3)

        :return: CEDᵢⱼ(r), correlation energy density
        :rtype: numpy.ndarray of shape (1,Mstate,Mstate,Ncoord)
           CED[0,i,j,r] is the correlation energy density between the
           electronic states i and j at position coords[r,:].
           There is only a single spin component, because the correlation
           energy depends on the total electron density and the spin polarization.
        """
        # number of grid points
        ncoord = coords.shape[0]
        # number of electronic states in the subspace
        nstate = msmd.number_of_states

        # Evaluate D(r) on the integration grid.
        D, _, _ = msmd.evaluate(coords)

        # total density D = Dᵅ(r) + Dᵝ(r)
        total_density = D[0,...] + D[1,...]
        # and spin polarization ΔD = Dᵅ(r) - Dᵝ(r)
        spin_polarization = D[0,...] - D[1,...]
        # WARNING: This functional only evaluates the paramagnetic part of the correlation
        #   energy, the spin polarization is assumed to be zero. For closed-shell molecules
        #   the diagonal elements of the matrix density are not spin-polarized, however
        #   the transition densities usually have a large spin-polarization.
        #   The correlation energy of the uniform electron gas is negative. For medium
        #   and low densities (0.5 ≤ rₛ ≤ 10.0), the ratio of ferromagnetic to paramagnetic
        #   correlation energy is approximately 1.1. So by neglecting the difference between
        #   paramagnetic and ferromagnetic correlation, we incur and error of approximately 10%.

        # correlation-energy density
        # CED[D]ᵢⱼ(r) = a log(Id + b₁ D(r)¹ᐟ³ + b₂ D(r)²ᐟ³ )
        # The array CED has only on spin dimension, because the functional operates
        # on the total spin density (up + down).
        CED = numpy.zeros((1,nstate,nstate,ncoord))

        for r in range(0, ncoord):
            # Compute eigenvalues Λ and eigenvectors U of the symmetric matrix
            # D = Dᵅ(r) + Dᵝ(r).
            L, U = numpy.linalg.eigh(total_density[:,:,r])
            # Numerical rounding errors might produce tiny, negative eigenvalues instead of 0.
            assert numpy.all(L > -1.0e-12), "Eigenvalues of matrix density D are expected to be positive."

            # The paramagnetic (spin=0) correlation energy function is applied to the eigenvalues
            #   CED[D](r) = U(r) εᶜ(Λ(r)) Uᵀ(r)
            ced_eigenvalues = LDACorrelationLikeFunctional.correlation_energy_density(abs(L), spin=0)
            CED[0,:,:,r] = numpy.einsum('ia,a,ja->ij', U, ced_eigenvalues, U)

        return CED
