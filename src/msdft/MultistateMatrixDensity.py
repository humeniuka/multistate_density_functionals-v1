#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The multistate matrix density D(r) for a subspace of N-electronic states an N x N
matrix with the state densities on the diagonal and the transition densities on the
off-diagonal.
"""
from abc import ABC, abstractmethod

import numpy
import scipy.linalg
import scipy.special

from pyscf.dft import numint
import pyscf.ci
import pyscf.fci
import pyscf.mcscf
import pyscf.scf
import pyscf.tddft


class MultistateMatrixDensity(ABC):
    def __init__(
            self,
            mol,
            eigenenergies,
            density_matrices):
        """
        This class holds the multistate matrix density and can evaluate
        D(r), ∇D(r) and ∇²D(r) on a grid.

        This is the base class, derived classes have to implement their own __init__
        functions to compute the (transition) density matrices and then call
        super().__init__(mol, density_matrices).

        :param mol: molecule with atomic coordinates, basis set and spin
        :type mol: pyscf.gto.Mole

        :param eigenenergies:
           eigenenergies[i] is the total energy of state i. Since the 2-particle density
           matrix is not available, the electron repulsion is calculated as the difference
           of the eigenenergies and the other terms in the Hamiltonian.
        :type eigenenergies: numpy.ndarray of shape (nstate,)

        :param density_matrices:
           density_matrices[spin,i,j,:,:] is the 1-particle (transition) density matrix
           between the electronic states i and j in the AO basis.
        :type density_matrices: numpy.ndarray of shape (2,nstate,nstate,nao,nao)
           nstate - number of electronic states
           nao - number of atomic orbitals
        """
        # Save molecule with AO basis.
        self.mol = mol
        # Check the dimensions of the (transition) density matrix.
        nspin, nstate1, nstate2, nao1, nao2 = density_matrices.shape
        assert nspin == 2, "Density matrix needs components for spin-up and spin-down."
        assert nstate1 == nstate2, "Matrix density has to be square"
        assert nao1 == nao2, "AO density matrix has to be square"
        assert len(eigenenergies) == nstate1, "len(eigenenergies) has to equal number of states"

        # Number of electronic states.
        self.number_of_states = nstate1
        # Save total energy, which includes nuclei-nuclei repulsion, kinetic energy,
        # nuclei-electrons attraction and electrons-electrons repulsion.
        self.eigenenergies = eigenenergies
        # Save (transition) density matrices.
        self.density_matrices = density_matrices

    def exact_1e_operator(self, intor='int1e_kin'):
        """
        For testing purposes the matrix of one-electron operators in the
        basis of the electronic states is calculated by contracting the
        (transition) density matrices in the AO basis with the AO integrals
        of the operator:

          Oᵢⱼ = <Ψᵢ|∑ₙ oₙ|Ψⱼ>

              = sum_{a,b} P^{i,j}_{a,b} <a|o|b>

        where i,j enumerate many-electron states, a,b are AOs and P^{i,j}_{a,b}
        is the (transition) density between the states i and j in the AO basis.

        :param intor: Name of the 1-electron integrals, e.g. 'int1e_kin' for
           the kinetic energy.
        :type intor: str

        :return matrix_elements: The matrix elements of the operator in the
           basis of the many-electron states in the subspace.
        :rtype matrix_elements: numpy.ndarray of shape (nstate,nstate)
        """
        integrals_1e_ao = self.mol.intor_symmetric(intor)
        matrix_elements = numpy.einsum(
            'ab,sijab->ij',
            integrals_1e_ao,
            self.density_matrices)

        return matrix_elements

    def exact_kinetic_energy(self):
        """
        compute the exact matrix elements of the kinetic energy operator

          Tᵢⱼ = <Ψᵢ|∑ₙ ∇²ₙ|Ψⱼ>

        :return kinetic_matrix: The matrix elements of the kinetic energy
           in the basis of the many-electron states in the subspace.
        :rtype kinetic_matrix: numpy.ndarray of shape (nstate,nstate)
        """
        return self.exact_1e_operator(intor='int1e_kin')

    def exact_coulomb_energy(self):
        """
        Compute the Coulomb integrals for all possible combinations of
        (transition) densities using the exact integrals between the Gaussian
        atomic orbitals.

          C[i,j,k,l] = ∫∫' Dᵢⱼ(r) Dₖₗ(r') /|r-r'|

                     = sum_{a,b,c,d} P^{i,j}_{a,b} (ab|cd) P^{k,l}_{c,d}

        where the (transition) density is expanded in the AO basis.

          Dᵢⱼ(r) = sum_{a,b} P^{i,j}_{a,b} χ_a(r) χ_b(r)

        :return coulomb_integrals:
           Coulomb integrals between (transition) densities
        :rtype coulomb_integrals:
           numpy.ndarray of shape (nstate,nstate,nstate,nstate)
        """
        nstate = self.number_of_states
        # Electron repulsion integrals (ab|cd)
        integrals_eri = self.mol.intor('int2e')
        # sum over spin
        dm_spin_trace = self.density_matrices[0,...] + self.density_matrices[1,...]

        # All combinations of Coulomb interactions between (transition densities)
        # D_{i,j}(r) and D_{k,l}(r)
        coulomb_integrals = numpy.einsum(
            'ijab,abcd,klcd->ijkl',
            dm_spin_trace,
            integrals_eri,
            dm_spin_trace)

        return coulomb_integrals

    def hartree_matrix_product(self):
        """
        compute the Hartree product of a matrix density with itself,

            J[D(r)]ᵢⱼ = 1/2 ∑ₖ ∫∫' Dᵢₖ(r) Dₖⱼ(r')/|r-r'|,

        using the representation of D(r) in the Gaussian AO basis,

            D(r)ᵢⱼ = sum_{a,b} P^{i,j}_{a,b} χ_a(r) χ_b(r),

        and the two-electron integrals (ab|cd),

            J[D(r)]ᵢⱼ = 1/2 ∑ₖ ∑_{a,b,c,d} P^{i,k}_{a,b} P^{k,j}_{c,d} (ab|cd)

        The contraction with the two-electron integrals does not require storing
        all integrals in memory.

        :return hartree_like_matrix: The Hartree-like matrix Jᵢⱼ in the subspace
           of the electronic states i,j=1,...,nstate
        :rtype hartree_like_matrix: numpy.ndarray of shape (nstate,nstate)
        """
        nspin, nstate, nstate, nao, nao = self.density_matrices.shape
        # sum over spin
        dm_spin_trace = self.density_matrices[0,...] + self.density_matrices[1,...]

        # According to the above doc-string, the contraction string should be
        # 'abcd,cd->ab', get_jk(...) requires that the contraction string contains
        # the indices ijkl.
        # Compute the electrostatic potential of the (transition) density matrices.
        # V[k,j,a,b] = ∑_{c,d} P^{k,j}_{c,d} (ab|cd)
        potential_list = pyscf.scf.jk.get_jk(
            self.mol,
            dm_spin_trace.reshape((nstate*nstate,nao,nao)),
            ['ijkl,kl->ij']*nstate*nstate)
        V = numpy.array(potential_list).reshape((nstate,nstate,nao,nao))
        # Contract density matrices with electrostatic potential.
        # J[D(r)]ᵢⱼ = 1/2 ∑ₖ ∑_{a,b} P^{i,k}_{a,b} V[k,j,a,b]
        hartree_like_matrix = 0.5 * numpy.einsum(
            'ikab,kjab->ij',
            dm_spin_trace,
            V)

        return hartree_like_matrix

    def exact_electron_repulsion(self):
        """
        Compute the matrix elements of the electron-repulsion operator between
        eigenfunctions Ψᵢ and Ψⱼ,

          Cᵢⱼ = ∫dx1 ∫dx2...∫dxn Ψ*ᵢ(x1,x2,...,xn) ∑ᵦ<ᵧ 1/|rᵦ-rᵧ| Ψⱼ(x1,x2,...,xn),

        exactly. Since the eigenfunctions diagonalize the Hamiltonian, the
        electron repulsion can be obtained from the eigenenergies E, the nuclear
        repulsion energy N, the kinetic energy T and the external potential
        (electron-nuclear attraction) V,

          Cᵢⱼ = (Eᵢ - N) δᵢⱼ - Tᵢⱼ - Vᵢⱼ

        :return repulsion_matrix: The matrix elements of the electron repulsion
           in the basis of the many-electron states in the subspace.
        :rtype repulsion_matrix: numpy.ndarray of shape (nstate,nstate)
        """
        # T
        kinetic_matrix = self.exact_1e_operator(intor='int1e_kin')
        # V
        nuclear_matrix = self.exact_1e_operator(intor='int1e_nuc')
        # (Eᵢ - N) δᵢⱼ
        electronic_energies = numpy.diag(self.eigenenergies - self.mol.energy_nuc())
        # Cᵢⱼ = (Eᵢ - N) δᵢⱼ - Tᵢⱼ - Vᵢⱼ
        electron_repulsion_matrix = electronic_energies - kinetic_matrix - nuclear_matrix

        return electron_repulsion_matrix

    def evaluate(self, coords):
        """
        evaluate the multistate matrix density D(r), its gradient ∇D(r)
        and its Laplacian ∇²D(r) on a grid.

        Mstate is the number of electronic states
        Ncoord is the number of grid points.

        :param coords: The Cartesian coordinates of the grid r
        :type coords: numpy.ndarray of shape (Ncoord,3)

        :return: D, grad_D, lapl_D
        :rtype: tuple of numpy.ndarray
          `D` has shape (2,Mstate,Mstate,Ncoord), D[s,i,j,c] is the (transition) density matrix
            for electrons with spin projection s=0 (up) or s=1 (down) evaluated at the grid point coords[c,:]
          `grad_D` has shape (2,Mstate,Mstate,3,Ncoord), grad_D[s,i,j,xyz,c] is the first-order
            derivative dD_ij(r)/dq (q=0(x), 1(y), 2(z)) evaluated at the grid point coords[c,:]
          `lapl_D` has shape (2,Mstate,Mstate,Ncoord), D[s,i,j,c] is the Laplacian of the
            (transition) density matrix for electrons with spin projection s=0 (up) or s=1 (down)
            evaluated at the grid point coords[c,:]
        """
        # number of grid points
        ncoord = coords.shape[0]
        # number of electronic states
        nstate = self.number_of_states
        # number of spins (up and down)
        nspin = 2

        # Create empty arrays for return values.
        D = numpy.zeros((nspin,nstate,nstate,ncoord))
        grad_D = numpy.zeros((nspin,nstate,nstate,3,ncoord))
        lapl_D = numpy.zeros((nspin,nstate,nstate,ncoord))

        # Evaluate atomic orbitals 𝛘ₐ(r) on the grid.
        # The orbital values and their gradients and Laplacian are returned in a single
        # array of shape (10,ncoord,norb).
        ao_value_all = numint.eval_ao(self.mol, coords, deriv=2)
        # value AO(r)
        ao_value = ao_value_all[0,:,:]
        # gradient d(AO)/dx, d(AO)/dy, d(AO)/dz
        grad_ao_value = ao_value_all[1:4,:,:]
        # Laplacian ∇²(AO)(r) = d^2(AO)/dx^2 + d^2(AO)/dy^2 + d^2(AO)/dz^2
        lapl_ao_value = ao_value_all[4,:,:] + ao_value_all[7,:,:] + ao_value_all[9,:,:]

        # Evaluate the matrix density functions on the grid.
        for spin in range(0, nspin):
            for i in range(0, nstate):
                for j in range(0, nstate):
                    # (transition) density in AO basis.
                    dao_ij = self.density_matrices[spin,i,j,:,:]
                    D[spin,i,j,:] = numpy.einsum('ab,ra,rb->r', dao_ij, ao_value, ao_value)
                    grad_D[spin,i,j,:,:] = (
                        numpy.einsum('ab,gra,rb->gr', dao_ij, grad_ao_value, ao_value) +
                        numpy.einsum('ab,ra,grb->gr', dao_ij, ao_value, grad_ao_value))

                    # ∇²D(r) = sum_{a,b} P_{a,b} [ (∇²𝛘*_a)(𝛘_b) + 2 (∇𝛘_a)·(∇𝛘_b) + (𝛘_a)(∇²𝛘*_b) ]
                    lapl_D[spin,i,j,:] = (
                        numpy.einsum('ab,ra,rb->r', dao_ij, lapl_ao_value, ao_value) +
                        2*numpy.einsum('ab,gra,grb->r', dao_ij, grad_ao_value, grad_ao_value) +
                        numpy.einsum('ab,ra,rb->r', dao_ij, ao_value, lapl_ao_value)
                        )

        return D, grad_D, lapl_D

    def evaluate_derivatives(self, coords, deriv=2):
        """
        evaluate the derivatives of the multistate matrix density D(r) on a grid.

        The partial derivatives of order n=0,...,`deriv` are calculated along
        the x-, y- and z-axes:

          ∂ⁿ/∂xⁿ D(x,y,z), ∂ⁿ/∂yⁿ D(x,y,z) and ∂ⁿ/∂zⁿ D(x,y,z)

        Mixed derivatives such as ∂²/∂x∂y D(x,y,z) are left out.

        Mstate is the number of electronic states
        Ncoord is the number of grid points.

        :param coords: The Cartesian coordinates of the grid r
        :type coords: numpy.ndarray of shape (Ncoord,3)

        :param deriv: maximum order of derivatives
        :type deriv: int >= 0

        :return: D_deriv
        :rtype: numpy.ndarray of shape (2,Mstate,Mstate,3,deriv+1,Ncoord)
          D_deriv[s,i,j,:,n,c] are the order n partial derivatives of the the (i,j)
          element of the (transition) matrix density Dˢᵢⱼ(x,y,z) with spin projection s (0=up, 1=down),
          [∂ⁿ/∂xⁿ Dˢᵢⱼ, ∂ⁿ/∂yⁿ Dˢᵢⱼ, ∂ⁿ/∂zⁿ Dˢᵢⱼ], evaluated at the grid point (x,y,z) = coords[c,:].

          Note that D_deriv[s,i,j,:,0,c] (the 0-order derivatives) is just the matrix density Dˢᵢⱼ(r)
          repeated for x, y and z.
        """
        # number of grid points
        ncoord = coords.shape[0]
        # number of electronic states
        nstate = self.number_of_states
        # number of spins (up and down)
        nspin = 2

        # Evaluate atomic orbitals 𝛘ₐ(r) on the grid.
        # The orbital values and their derivatives are returned in a single array.
        ao_derivs = numint.eval_ao(self.mol, coords, deriv=deriv)
        # The partial derivatives are ordered as follows:
        #
        # deriv=1:   ['x', 'y', 'z']
        # deriv=2:   ['xx', 'xy', 'xz', 'yy', 'yz', 'zz']
        # deriv=3:   ['xxx', 'xxy', 'xxz', 'xyy', 'xyz', 'xzz', 'yyy', 'yyz', 'yzz', 'zzz']
        #
        # The ordering for the general case deriv=n can be found with
        # >>> list(map(lambda s: ''.join(s), list(itertools.combinations_with_replacement('xyz', n))))
        #

        # We are only interested in the partial derivatives along the same axis,
        # i.e. x, xx, xxx, ...
        #      y, yy, yyy, ...
        #      z, zz, zzz, ...
        # `index_into_derivatives[k,:]` gives the indeces at with the derivatives
        # ∂ᵏ/∂xᵏ, ∂ᵏ/∂yᵏ and ∂ᵏ/∂zᵏ can be found in the array of partial derivatives returned by pyscf.
        index_into_derivatives = numpy.zeros((deriv+1, 3), dtype=int)
        offset = 0
        for k in range(0, deriv+1):
            # The derivatives of degree k contribute k*(k+1)/2 components in `ao_derivs`
            offset += (k*(k+1))//2
            # x-derivatives ∂ᵏ/∂xᵏ 𝛘 are stored at the index offset+0 of `ao_derivs`
            index_into_derivatives[k, 0] = offset + 0
            # y-derivatives ∂ᵏ/∂yᵏ 𝛘 are stored at the index offset+k*(k+1)/2 of `ao_derivs`
            index_into_derivatives[k, 1] = offset + (k*(k+1))//2
            # z-derivatives ∂ᵏ/∂zᵏ 𝛘 are stored at the index offset+(k+1)*(k+2)/2-1 of `ao_derivs`
            index_into_derivatives[k, 2] = offset + ((k+1)*(k+2))//2-1

        # Allocate memory for the derivatives
        D_derivs = numpy.zeros((nspin,nstate,nstate,3,deriv+1,ncoord))

        # Evaluate the derivatives of the matrix density Dˢᵢⱼ(x,y,z) on the grid.
        # Loop over spins
        for spin in range(0, nspin):
            # Loop over electronic states (bra)
            for i in range(0, nstate):
                # Loop over electronic state (ket)
                for j in range(0, nstate):
                    # (transition) density Pⁱʲ_{a,b} in AO basis.
                    dao_ij = self.density_matrices[spin,i,j,:,:]
                    # Since Dᵢⱼ(r) = ∑_{a,b} Pⁱʲ_{a,b} 𝛘a(r) 𝛘b(r),
                    # the derivatives of D(r) are obtained by applying the product rule repeatedly,
                    # which gives rise to Leibniz' rule:
                    #   ∂ⁿ/∂xⁿ Dᵢⱼ = ∑_{k=0}^n binom(n,k)
                    #                    ∑_{a,b} Pⁱʲ_{a,b} [∂ⁿ⁻ᵏ/∂xⁿ⁻ᵏ 𝛘a] [∂ᵏ/∂xᵏ 𝛘b]
                    for n in range(0, deriv+1):
                        # ∑_{k=0}^n
                        for k in range(0, n+1):
                            # Loop over x,y and z axis.
                            for xyz in [0,1,2]:
                                D_derivs[spin,i,j,xyz,n,:] += (
                                    scipy.special.binom(n,k) *
                                    # ∑_{a,b}
                                    numpy.einsum('ab,ra,rb->r',
                                        # Pⁱʲ_{a,b}
                                        dao_ij,
                                        # ∂ⁿ⁻ᵏ/∂xⁿ⁻ᵏ 𝛘a(r)
                                        ao_derivs[index_into_derivatives[n-k, xyz],:,:],
                                        # ∂ᵏ/∂xᵏ 𝛘b(r)
                                        ao_derivs[index_into_derivatives[k,   xyz],:,:])
                                    )

        return D_derivs

    def kinetic_energy_density(
            self,
            coords : numpy.ndarray):
        """
        The kinetic energy density

           KEDᵢⱼ(r) = <Ψᵢ|-1/2 ∑ₙ δ(r-rₙ) ∇ₙ²|Ψⱼ>

        can be computed in two ways:

          KEDᵢⱼ(r) = -1/2 ∑_a ∑_b Dᵢⱼ(a,b) 𝛘_a(r) ∇²𝛘_b(r)   (Laplacian)

        or as

          KEDᵢⱼ(r) = 1/2 ∑_a ∑_b Dᵢⱼ(a,b) ∇𝛘_a(r) · ∇𝛘_b(r)   (scalar product of gradients)

        Both kinetic energy densities integrate to the same kinetic energy matrix
        (see DOI:10.1063/1.1565316) as they only differ by a term of the form
        1/4 ∇²D(r) that vanishes after integrating over all space.

        :param coords: The Cartesian positions at which the kinetic energy
           density is calculated.
        :type coords: numpy.ndarray of shape (Ncoord,3)

        :return: (KEDlapᵢⱼ(r), KEDggᵢⱼ(r))
           kinetic energy densities computed in the two ways
        :rtype: two numpy.ndarray's of shape (2,Mstate,Mstate,Ncoord) each
           KED[s,i,j,r] is the kinetic energy density
        """
        # number of grid points
        ncoord = coords.shape[0]
        # number of electronic states
        nstate = self.number_of_states
        # number of spins (up and down)
        nspin = 2

        # Create empty arrays for the kinetic energy densities.
        KED_laplacian = numpy.zeros((nspin,nstate,nstate,ncoord))
        KED_gradgrad = numpy.zeros((nspin,nstate,nstate,ncoord))

        # Evaluate atomic orbitals 𝛘ₐ(r) on the grid.
        # The orbital values and their gradients are returned in a single
        # array of shape (4,ncoord,norb).
        ao_value_all = numint.eval_ao(self.mol, coords, deriv=2)
        # value AO(r)
        ao_value = ao_value_all[0,:,:]
        # gradient d(AO)/dx, d(AO)/dy, d(AO)/dz
        grad_ao_value = ao_value_all[1:4,:,:]
        # Laplacian ∇²(AO)(r) = d^2(AO)/dx^2 + d^2(AO)/dy^2 + d^2(AO)/dz^2
        lapl_ao_value = ao_value_all[4,:,:] + ao_value_all[7,:,:] + ao_value_all[9,:,:]

        # Evaluate the kinetic energy density
        for spin in range(0, nspin):
            for i in range(0, nstate):
                for j in range(0, nstate):
                    # (transition) density in AO basis.
                    dao_ij = self.density_matrices[spin,i,j,:,:]

                    # using the Laplacian of the orbitals
                    KED_laplacian[spin,i,j,:] = -0.5 * numpy.einsum(
                        'ab,ra,rb->r',
                        dao_ij, ao_value, lapl_ao_value)
                    # or using the gradients of the orbitals.
                    KED_gradgrad[spin,i,j,:] = 0.5 * numpy.einsum(
                        'ab,dra,drb->r',
                        dao_ij, grad_ao_value, grad_ao_value)

        return KED_laplacian, KED_gradgrad

    @staticmethod
    @abstractmethod
    def create_matrix_density(mol, nstate=4):
        """
        Compute the multistate matrix density for the lowest few excited states
        of a small molecule using the respective electronic structure method
        of the derived class.

        :param mol: A test molecule
        :type mol: gto.Mole

        :param nstate: number of excited states to calculate
        :type nstate: positive int

        :return: multistate matrix density
        :rtype: :class:`~.MultistateMatrixDensity`
        """
        pass

    def align_phases(
            self,
            msmd_ref):
        """
        Wavefunctions are only uniquely defined up to a global phase. Multiplying each
        eigenstate Ψᵢ by an arbitrary sign σᵢ = ±1 does not change any observable.
        However, the signs of the off-diagonal matrix elements of the matrix density are
        changed:

             Ψᵢ → σᵢΨᵢ  leads to  Dᵢⱼ(r) → σᵢσⱼDᵢⱼ(r)

        Eigensolvers produce essentially eigenvectors with arbitrary phases. In order to
        have the matrix density change continuously as a function of some external parameter
        such as the nuclear coordinates, the phases between neighbouring D'ᵢⱼ and Dᵢⱼ have
        to be aligned such that ||D'ᵢⱼ - σᵢσⱼ Dᵢⱼ|| is minimized.

        This function finds the phases σᵢ that align the matrix density Dᵢⱼ of `self`
        with a reference density D'ᵢⱼ of `msdm_ref`.
        The signs are applied in place to the matrix density Dᵢⱼ.

        :param msmd_ref: reference matrix density
        :type msmd_ref: :class:`~.MultistateMatrixDensity`
        """
        # Evaluate the matrix densities on a coarse grid
        grids = pyscf.dft.gen_grid.Grids(self.mol)
        grids.level = 1
        grids.build()
        # D(r)
        D, _, _ = self.evaluate(grids.coords)
        # The reference D'(r)
        D_ref, _, _ = msmd_ref.evaluate(grids.coords)
        # Similarity between D and D' is measured by the
        # scalar product <D,D'> = ∫ Dᵢⱼ(r) D'ᵢⱼ(r) dr
        overlap = numpy.einsum('r,sijr,sijr->ij', grids.weights, D, D_ref)
        # The similarity is normalized by the norm squared of reference
        # ||D'||² = <D',D'> = ∫ D'ᵢⱼ(r) D'ᵢⱼ(r) dr
        norm_squared = numpy.einsum('r,sijr,sijr->ij', grids.weights, D_ref, D_ref)
        # If D and D' are similar and have the same phases (σᵢ=1),
        # the matrix Sᵢⱼ = <D,D'>/<D',D'> = σᵢσⱼ should be a matrix that has ones everywhere.
        similarity = overlap / norm_squared
        # To extract the vector of phases σᵢ from the product Sᵢⱼ = σᵢσⱼ, an eigenvalue
        # decomposition is performed. If D and D' differ only by the signs, there should
        # be only a single non-zero eigenvalue and the corresponding eigenvector is just σᵢ.
        eigvals, eigvecs = scipy.linalg.eigh(similarity)
        # The last eigenvector.
        signs = numpy.sign(eigvecs[:,-1]).astype(int)
        # The largest eigenvalue should be close to `number_of_states` and all
        # other eigenvalues should be approximately zero.

        # Apply the sign to the one-particle (transition) density matrices.
        self.density_matrices = numpy.einsum('i,j,sijab->sijab', signs, signs, self.density_matrices)

    def _zero_transition_densities(self):
        """
        Set the off-diagonal elements for iof the matrix density D(r) to zero,
        i.e. Dᵢⱼ(r) = 0 for i ≠ j.

        This underscore method only exists for debugging purposes.
        """
        # number of electronic states
        nstate = self.number_of_states
        # Loop over states
        for i in range(0, nstate):
            for j in range(0, nstate):
                if i != j:
                    # zero out transition density in AO basis.
                    self.density_matrices[:,i,j,:,:] *= 0.0



def _density_matrix_mo2ao(dm_mo, mo_coeff):
    """
    transform a density matrix in the MO basis in the AO basis

        P^AO_{a,b}   = sum_{m,n} C*_{a,m} P^MO_{m,n} C_{b,n}

    a,b enumerate atomic orbitals, m,n enumerate molecular orbitals
    and C_{a,m} are the self-consistent field MO coefficients.

    :param dm_mo: density matrix in MO basis
    :type dm_mo: numpy.ndarray of shape (nmo,nmo)

    :param mo_coeff: molecular orbital coefficients
    :type mo_coeff: numpy.ndarray of shape (nao,nmo)

    :return dm_ao: density matrix in AO basis
    :rtype dm_ao: numpy.ndarray of shape (nao,nao)
    """
    nao, nmo = mo_coeff.shape
    assert dm_mo.shape == (nmo,nmo)
    dm_ao = numpy.einsum(
        'am,mn,bn->ab',
        mo_coeff, dm_mo, mo_coeff)
    return dm_ao


class MultistateMatrixDensityFCI(MultistateMatrixDensity):
    def __init__(
            self,
            mol,
            rhf,
            fci,
            fcivecs):
        """
        This class holds the multistate matrix density and can evaluate
        D(r), ∇D(r) and ∇²D(r) on a grid.
        The state densities and transition densities are constructed from
        a full configuration interaction calculation with pyscf.

        :param mol: molecule with atomic coordinates, basis set and spin
        :type mol: pyscf.gto.Mole

        :param rhf: restricted self-consistent field solution with molecular orbitals
        :type rhf: pyscf.scf.RHF

        :param fci: full configuration interaction solved
        :type fci: pyscf.fci.FCI

        :param fcivecs: list of solution vectors of the CI problem for
          each electronic state in the subspace
        :type fcivecs: list of numpy.ndarray
        """
        # number of atomic orbitals and molecular orbitals
        nao, nmo = rhf.mo_coeff.shape

        # If there is only a single state, the energies and FCI vectors
        # are not stored as a list.
        if hasattr(fci.e_tot, '__len__'):
            eigenenergies = fci.e_tot[:len(fcivecs)]
        else:
            eigenenergies = numpy.array([fci.e_tot])
            fcivecs = [fcivecs]

        # number of electronic states
        nstate = len(fcivecs)
        # Compute the (transition) density matrices in the AO basis.
        nspin = 2
        density_matrices = numpy.zeros((nspin,nstate,nstate,nao,nao))
        for i in range(0, nstate):
            for j in range(0, nstate):
                if i == j:
                    # 1-particle density matrix of state i in MO basis
                    dm1a, dm1b = fci.make_rdm1s(fcivecs[i], nmo, mol.nelec)
                    # for spin-up
                    density_matrices[0,i,i,:,:] = _density_matrix_mo2ao(dm1a, rhf.mo_coeff)
                    # for spin-down
                    density_matrices[1,i,i,:,:] = _density_matrix_mo2ao(dm1b, rhf.mo_coeff)
                else:
                    # 1-particle transition density matrix
                    # between electronic states i and j.
                    tdm1a, tdm1b = fci.trans_rdm1s(fcivecs[i], fcivecs[j], nmo, mol.nelec)
                    # for spin-up
                    density_matrices[0,i,j,:,:] = _density_matrix_mo2ao(tdm1a, rhf.mo_coeff)
                    # for spin-down
                    density_matrices[1,i,j,:,:] = _density_matrix_mo2ao(tdm1b, rhf.mo_coeff)

        # Initialize base class.
        super().__init__(mol, eigenenergies, density_matrices)

    @staticmethod
    def create_matrix_density(mol, nstate=4, spin_symmetry=True, raise_error=True):
        """
        Compute the multistate matrix density for the lowest few excited states
        of a small molecule using full configuration interaction.

        :param mol: A test molecule
        :type mol: gto.Mole

        :param nstate: number of electronic states to calculate
        :type nstate: positive int

        :param spin_symmetry: use of spin symmetry in the CI calculation
        :type spin_symmetry: bool

        :param raise_error: Raise an error if the CI space is smaller
          than the number of requested states `nstate`.
        :type raise_error: bool

        :return: multistate matrix density
        :rtype: :class:`~.MultistateMatrixDensity`
        """
        assert nstate > 0
        hf = pyscf.scf.RHF(mol)
        # supress printing of SCF energy
        hf.verbose = 0
        # compute self-consistent field
        hf.kernel()

        # singlet=True enables the use of spin symmetry in the CI calculation.
        fci = pyscf.fci.FCI(mol, hf.mo_coeff, singlet=spin_symmetry)
        # Solve for the lower few electronic states.
        fci.nroots = nstate
        fci_energies, fcivecs = fci.kernel()

        if hasattr(fci.e_tot, '__len__'):
            nstate_available = len(fci.e_tot)
        else:
            nstate_available = 1

        if nstate_available < nstate and raise_error:
            raise RuntimeError(
                f"Size of full CI space ({nstate_available}) is smaller "
                f"than number of requested states ({nstate})")

        msmd = MultistateMatrixDensityFCI(mol, hf, fci, fcivecs)

        return msmd


class MultistateMatrixDensityCISD(MultistateMatrixDensity):
    def __init__(
            self,
            mol,
            rhf,
            cisd):
        """
        This class holds the multistate matrix density and can evaluate
        D(r), ∇D(r) and ∇²D(r) on a grid.
        The state densities and transition densities are constructed from
        a configuration interaction calculation with singles and doubles (CISD)
        with pyscf.

        :param mol: molecule with atomic coordinates, basis set and spin
        :type mol: pyscf.gto.Mole

        :param rhf: restricted self-consistent field solution with molecular orbitals
        :type rhf: pyscf.scf.RHF

        :param cisd: solved CISD problem
        :type cisd: pyscf.ci.CISD
        """
        # number of atomic orbitals and molecular orbitals
        nao, nmo = rhf.mo_coeff.shape

        # If there is only a single state, the energies and CISD vectors
        # are not stored as a list.
        if hasattr(cisd.e_tot, '__len__'):
            eigenenergies = cisd.e_tot
            cisd_vectors = cisd.ci
        else:
            eigenenergies = numpy.array([cisd.e_tot])
            cisd_vectors = [cisd.ci]

        # number of electronic states
        nstate = len(cisd_vectors)

        # Compute the (transition) density matrices in the AO basis.
        nspin = 2
        density_matrices = numpy.zeros((nspin,nstate,nstate,nao,nao))
        for i in range(0, nstate):
            for j in range(0, nstate):
                if i == j:
                    # 1-particle density matrix of state i in MO basis
                    if isinstance(cisd, pyscf.ci.cisd.RCISD):
                        dm = cisd.make_rdm1(
                            cisd_vectors[i], nmo=nmo, nocc=sum(mol.nelec)//2)
                        # CISD.make_rdm1 returns the spin-traced density matrix
                        dm1a, dm1b = 0.5*dm, 0.5*dm
                    elif isinstance(cisd, pyscf.ci.ucisd.UCISD):
                        # UCSID.make_rdm1 returns density matrices for spin-up and spin-down.
                        dm1a, dm1b = cisd.make_rdm1(
                            cisd_vectors[i], nmo=(nmo,nmo), nocc=mol.nelec)
                    else:
                        raise TypeError(
                            "Argument `cisd` has to be of type `pyscf.ci.cisd.RCISD` "
                            " or `pyscf.ci.cisd.UCISD`")
                    # for spin-up
                    density_matrices[0,i,i,:,:] = _density_matrix_mo2ao(dm1a, rhf.mo_coeff)
                    # for spin-down
                    density_matrices[1,i,i,:,:] = _density_matrix_mo2ao(dm1b, rhf.mo_coeff)
                else:
                    # 1-particle transition density matrix
                    # between electronic states i and j.
                    if isinstance(cisd, pyscf.ci.cisd.RCISD):
                        tdm = cisd.trans_rdm1(
                            cisd_vectors[i], cisd_vectors[j], nmo=nmo, nocc=sum(mol.nelec)//2)
                        # CISD.trans_rdm1 returns the spin-traced transition density matrix.
                        tdm1a, tdm1b = 0.5*tdm, 0.5*tdm
                    elif isinstance(cisd, pyscf.ci.ucisd.UCISD):
                        # UCSID.trans_rdm1 returns transition density matrices
                        # for spin-up and spin-down.
                        tdm1a, tdm1b = cisd.trans_rdm1(
                            cisd_vectors[i], cisd_vectors[j], nmo=(nmo,nmo), nocc=mol.nelec)
                    else:
                        raise TypeError(
                            "Argument `cisd` has to be of type `pyscf.ci.cisd.RCISD` "
                            " or `pyscf.ci.cisd.UCISD`")
                    # for spin-up
                    density_matrices[0,i,j,:,:] = _density_matrix_mo2ao(tdm1a, rhf.mo_coeff)
                    # for spin-down
                    density_matrices[1,i,j,:,:] = _density_matrix_mo2ao(tdm1b, rhf.mo_coeff)

        # Initialize base class.
        super().__init__(mol, eigenenergies, density_matrices)

    @staticmethod
    def create_matrix_density(mol, nstate=4, raise_error=True):
        """
        Compute the multistate matrix density for the lowest few excited states
        of a small molecule using configuration interaction with single and double excitations
        from the Hartree-Fock reference determinant (CISD).

        :param mol: A test molecule
        :type mol: gto.Mole

        :param nstate: number of electronic states to calculate
        :type nstate: positive int

        :param raise_error: Raise an error if the CISD space is smaller
          than the number of requested states `nstate`.
        :type raise_error: bool

        :return: multistate matrix density
        :rtype: :class:`~.MultistateMatrixDensity`
        """
        assert nstate > 0
        hf = pyscf.scf.RHF(mol)
        # supress printing of SCF energy
        hf.verbose = 0
        # compute self-consistent field
        hf.kernel()
        # number of molecular orbitals
        nmo = hf.mo_coeff.shape[1]

        # The electronic structure is solved with the CISD method.
        cisd = pyscf.ci.CISD(hf)
        # Solve for the lowest `nstate` states.
        cisd.nstates = nstate
        # supress printing of CISD energies
        cisd.verbose = 0
        cisd.kernel()

        if hasattr(cisd.e_tot, '__len__'):
            nstate_available = len(cisd.e_tot)
        else:
            nstate_available = 1

        if nstate_available < nstate and raise_error:
            raise RuntimeError(
                f"Size of full CISD space ({nstate_available}) is smaller "
                f"than number of requested states ({nstate})")

        msmd = MultistateMatrixDensityCISD(mol, hf, cisd)

        return msmd


def _density_matrix_cas2ao(dm_active, nocc, ncas, mo_coeff, is_transition_dm):
    """
    transform a density matrix in the basis of active MOS into the AO basis

        P^AO_{a,b}   = sum_{m,n} C*_{a,m} P^MO_{m,n} C_{b,n}

    a,b enumerate atomic orbitals, m,n enumerate molecular orbitals
    and C_{a,m} are the self-consistent field MO coefficients.

    :param dm_active: density matrix in basis of active orbitals
    :type dm_active: numpy.ndarray of shape (ncas,ncas)

    :param nocc: number of occupied spin orbitals
    :type nocc: int > 0

    :param ncas: number of active spin orbitals
    :type ncas: int > 0

    :param mo_coeff: MO coefficients for all orbitals (active and inactive)
    :type mo_coeff: numpy.ndarray of shape (nao,nmo)

    :param is_transition_dm: Whether `dm_active` is a transition density
        matrix (True) or a density matrix for a single state (False)
    :type is_transition_dm: bool

    :return dm_ao: density matrix in AO basis
    :rtype dm_ao: numpy.ndarray of shape (nao,nao)
    """
    assert dm_active.shape == (ncas,ncas)
    nmo = mo_coeff.shape[1]
    # The (transition) density matrix in the active space of shape (ncas,ncas)
    # is enlarged to a density matrix of shape (nmo,nmo) by adding
    # the closed and the virtual blocks.
    # For density matrices:
    #    DM(MO) = diag(1,...,1) ⊗ DM(active) ⊗ diag(0,...,0)
    #               closed       active      virtual
    # For transition density matrices:
    #    TDM(MO) = diag(0,...,0) ⊗ DM(active) ⊗ diag(0,...,0)
    #               closed       active      virtual
    if is_transition_dm:
        dm_occupied = numpy.diag([0] * nocc)
    else:
        dm_occupied = numpy.diag([1] * nocc)
    dm_virtual = numpy.diag([0] * (nmo-(nocc+ncas)))
    # Density matrix in the MO basis
    dm_mo = scipy.linalg.block_diag(
        dm_occupied,
        dm_active,
        dm_virtual
    )
    # MO coefficients of active orbitals
    dm_ao = numpy.einsum(
        'am,mn,bn->ab',
        mo_coeff, dm_mo, mo_coeff)
    return dm_ao


class MultistateMatrixDensityCASCI(MultistateMatrixDensity):
    def __init__(
            self,
            mol,
            rhf,
            casci):
        """
        This class holds the multistate matrix density and can evaluate
        D(r), ∇D(r) and ∇²D(r) on a grid.
        The state densities and transition densities are constructed from
        a complete active space configuration interaction (CASCI) calculation
        with pyscf.

        :param mol: molecule with atomic coordinates, basis set and spin
        :type mol: pyscf.gto.Mole

        :param rhf: restricted self-consistent field solution with molecular orbitals
        :type rhf: pyscf.scf.RHF

        :param casci: solved CASCI problem
        :type casci: pyscf.mcscf.CASCI
        """
        # number of atomic orbitals
        nao = casci.mo_coeff.shape[0]

        # If there is only a single state, the energies and CI vectors
        # are not stored as lists.
        if hasattr(casci.e_tot, '__len__'):
            eigenenergies = casci.e_tot
            # CAS space FCI coefficients
            ci_vectors = casci.ci
        else:
            eigenenergies = numpy.array([casci.e_tot])
            # CAS space FCI coefficients
            ci_vectors = [casci.ci]

        # number of electronic states
        nstate = len(ci_vectors)

        # Compute the (transition) density matrices in the AO basis.
        nspin = 2
        density_matrices = numpy.zeros((nspin,nstate,nstate,nao,nao))
        for i in range(0, nstate):
            for j in range(0, nstate):
                if i == j:
                    # 1-particle density matrix of state i in MO basis
                    dm1a, dm1b = casci.fcisolver.make_rdm1s(
                        ci_vectors[i], casci.ncas, casci.nelecas)
                    # for spin-up
                    density_matrices[0,i,i,:,:] = _density_matrix_cas2ao(
                        dm1a, nocc=mol.nelec[0]-casci.nelecas[0], ncas=casci.ncas,
                        mo_coeff=casci.mo_coeff, is_transition_dm=False)
                    # for spin-down
                    density_matrices[1,i,i,:,:] = _density_matrix_cas2ao(
                        dm1b, nocc=mol.nelec[1]-casci.nelecas[1], ncas=casci.ncas,
                        mo_coeff=casci.mo_coeff, is_transition_dm=False)
                else:
                    # 1-particle transition density matrix
                    # between electronic states i and j.
                    tdm1a, tdm1b = casci.fcisolver.trans_rdm1s(
                        ci_vectors[i], ci_vectors[j], casci.ncas, casci.nelecas)
                    # for spin-up
                    density_matrices[0,i,j,:,:] = _density_matrix_cas2ao(
                        tdm1a, nocc=mol.nelec[0]-casci.nelecas[0], ncas=casci.ncas,
                        mo_coeff=casci.mo_coeff, is_transition_dm=True)
                    # for spin-down
                    density_matrices[1,i,j,:,:] = _density_matrix_cas2ao(
                        tdm1b, nocc=mol.nelec[1]-casci.nelecas[1], ncas=casci.ncas,
                        mo_coeff=casci.mo_coeff, is_transition_dm=True)

        # Initialize base class.
        super().__init__(mol, eigenenergies, density_matrices)

    @staticmethod
    def create_matrix_density(
            mol,
            nstate=4,
            ncas=None,
            nelecas=None,
            spin_symmetry=True,
            raise_error=True):
        """
        Compute the multistate matrix density for the lowest few excited states
        of a small molecule using complete active space configuration interaction (CASCI).

        :param mol: A test molecule
        :type mol: gto.Mole

        :param nstate: number of electronic states to calculate
        :type nstate: positive int

        :param ncas: number of active orbitals
        :type ncas: int > 0 or None to include all orbitals

        :param nelecas: number of active electrons
        :type nelecas: int > 0 or None to include all electrons

        :param spin_symmetry: use of spin symmetry in the CI calculation
          States with an undesired spin are shifted up in energy. It can therefore
          still happens that states with different spin appear in the spectrum.
        :type spin_symmetry: bool

        :param raise_error: Raise an error if the CASCI space is smaller
          than the number of requested states `nstate`.
        :type raise_error: bool

        :return: multistate matrix density
        :rtype: :class:`~.MultistateMatrixDensity`
        """
        assert nstate > 0
        hf = pyscf.scf.RHF(mol)
        # supress printing of SCF energy
        hf.verbose = 0
        # compute self-consistent field
        hf.kernel()
        # number of molecular orbitals
        nmo = hf.mo_coeff.shape[1]

        # If ncas and nelecas are None, the full active space is used.
        if ncas is None:
            ncas = nmo
        if nelecas is None:
            nelecas = sum(mol.nelec)

        # The number of active orbitals cannot be larger than the total number
        # of molecular orbitals.
        ncas = min(ncas, nmo)

        # The number of unique determinants that can be built by distributing
        # nelecas electrons over ncas orbitals is
        #  orbitals! / (electrons! (orbitals-electrons)!)
        nstate_available = int(
            # Number of possibilities to distribute the up electrons over the active orbitals.
            scipy.special.binom(ncas, nelecas - nelecas//2) *
            # Number of possibilities to distribute the down electrons over the active orbitals.
            scipy.special.binom(ncas, nelecas//2)
        )
        if nstate > nstate_available:
            if raise_error:
                raise RuntimeError(
                    f"Size of full CASCI space ({nstate_available}) is smaller "
                    f"than number of requested states ({nstate})")
            else:
                # Just reduce the number of states.
                nstate = nstate_available

        # The electronic structure is solved with the CASCI method.
        casci = pyscf.mcscf.CASCI(hf, ncas, nelecas)
        casci.fcisolver.nstates = nstate
        # Tighten convergence criteria,
        # otherwise we get slightly different results than with FCI.
        casci.fcisolver.conv_tol = 1e-12
        casci.fcisolver.max_cycle = 1000
        if spin_symmetry:
            # States with undesired spins are shifted up in energy.
            casci.fix_spin(shift=0.5)
        # supress printing of CASCI energies
        casci.verbose = 0

        # If the number of requested states is larger then the full CASCI
        # space, the calculation will fail with something like
        #  ValueError: kth(=<requested states>) out of bounds (<available states>)
        casci.kernel()

        msmd = MultistateMatrixDensityCASCI(mol, hf, casci)

        return msmd


class MultistateMatrixDensityCASSCF(MultistateMatrixDensity):
    def __init__(
            self,
            mol,
            casscf):
        """
        This class holds the multistate matrix density and can evaluate
        D(r), ∇D(r) and ∇²D(r) on a grid.
        The state densities and transition densities are constructed from
        a complete active space self-consistent field (CASSCF) calculation
        with pyscf.

        :param mol: molecule with atomic coordinates, basis set and spin
        :type mol: pyscf.gto.Mole

        :param casscf: solved CASSCF problem
        :type casscf: pyscf.mcscf.CASSCF
        """
        # number of atomic orbitals and molecular orbitals
        nao, nmo = casscf.mo_coeff.shape

        # If there is only a single state, the energies and CI vectors
        # are not stored as lists.
        if hasattr(casscf, 'e_states'):
            eigenenergies = casscf.e_states
            # CAS space CI coefficients
            ci_vectors = casscf.ci
        else:
            # If there is only a single state, only the CASSCF energy is stored.
            eigenenergies = numpy.array([casscf.e_tot])
            # CAS space CI coefficients
            ci_vectors = [casscf.ci]

        # number of electronic states
        nstate = len(ci_vectors)
        # Compute the (transition) density matrices in the AO basis.
        nspin = 2
        density_matrices = numpy.zeros((nspin,nstate,nstate,nao,nao))
        for i in range(0, nstate):
            for j in range(0, nstate):
                if i == j:
                    # 1-particle density matrix of state i in MO basis
                    dm1a, dm1b = casscf.fcisolver.trans_rdm1s(
                        ci_vectors[i], ci_vectors[i], casscf.ncas, casscf.nelecas)
                    # for spin-up
                    density_matrices[0,i,i,:,:] = _density_matrix_cas2ao(
                        dm1a, nocc=mol.nelec[0]-casscf.nelecas[0], ncas=casscf.ncas,
                        mo_coeff=casscf.mo_coeff, is_transition_dm=False)
                    # for spin-down
                    density_matrices[1,i,i,:,:] = _density_matrix_cas2ao(
                        dm1b, nocc=mol.nelec[1]-casscf.nelecas[1], ncas=casscf.ncas,
                        mo_coeff=casscf.mo_coeff, is_transition_dm=False)
                else:
                    # 1-particle transition density matrix
                    # between electronic states i and j.
                    tdm1a, tdm1b = casscf.fcisolver.trans_rdm1s(
                        ci_vectors[i], ci_vectors[j], casscf.ncas, casscf.nelecas)
                    # for spin-up
                    density_matrices[0,i,j,:,:] = _density_matrix_cas2ao(
                        tdm1a, nocc=mol.nelec[0]-casscf.nelecas[0], ncas=casscf.ncas,
                        mo_coeff=casscf.mo_coeff, is_transition_dm=True)
                    # for spin-down
                    density_matrices[1,i,j,:,:] = _density_matrix_cas2ao(
                        tdm1b, nocc=mol.nelec[1]-casscf.nelecas[1], ncas=casscf.ncas,
                        mo_coeff=casscf.mo_coeff, is_transition_dm=True)

        # Initialize base class.
        super().__init__(mol, eigenenergies, density_matrices)

    @staticmethod
    def create_matrix_density(
            mol,
            nstate=4,
            ncas=None,
            nelecas=None,
            spin_symmetry=True,
            raise_error=True):
        """
        Compute the multistate matrix density for the lowest few excited states
        of a small molecule using complete active space self-consistent field (CASSCF).

        :param mol: A test molecule
        :type mol: gto.Mole

        :param nstate: number of electronic states to calculate
        :type nstate: positive int

        :param ncas: number of active orbitals
        :type ncas: int > 0 or None to include all orbitals

        :param nelecas: number of active electrons
        :type nelecas: int > 0 or None to include all electrons

        :param spin_symmetry: use of spin symmetry in the CI calculation
          States with an undesired spin are shifted up in energy. It can therefore
          still happens that states with different spin appear in the spectrum.
        :type spin_symmetry: bool

        :param raise_error: Raise an error if the CASCI space is smaller
          than the number of requested states `nstate`.
        :type raise_error: bool

        :return: multistate matrix density
        :rtype: :class:`~.MultistateMatrixDensity`
        """
        assert nstate > 0
        hf = pyscf.scf.RHF(mol)
        # supress printing of SCF energy
        hf.verbose = 0
        # compute self-consistent field
        hf.kernel()
        # number of molecular orbitals
        nmo = hf.mo_coeff.shape[1]

        # If ncas and nelecas are None, the full orbital space is used.
        if ncas is None:
            ncas = nmo
        if nelecas is None:
            nelecas = sum(mol.nelec)

        # The number of active orbitals cannot be larger than the total number
        # of molecular orbitals.
        ncas = min(ncas, nmo)

        # The number of unique determinants that can be built by distributing
        # nelecas electrons over ncas orbitals is
        #  orbitals! / (electrons! (orbitals-electrons)!)
        nstate_available = int(
            # Number of possibilities to distribute the up electrons over the active orbitals.
            scipy.special.binom(ncas, nelecas - nelecas//2) *
            # Number of possibilities to distribute the down electrons over the active orbitals.
            scipy.special.binom(ncas, nelecas//2)
        )
        if nstate > nstate_available:
            if raise_error:
                raise RuntimeError(
                    f"Size of full CASSCF space ({nstate_available}) is smaller "
                    f"than number of requested states ({nstate})")
            else:
                # Just reduce the number of states.
                nstate = nstate_available

        casscf = pyscf.mcscf.CASSCF(hf, ncas, nelecas)
        if nstate > 1:
            # Weights for state-averaged CASSCF, all states have equal weights.
            weights = numpy.array([1.0/nstate] * nstate)
            casscf = casscf.state_average(weights)
        casscf.nstate = nstate
        if spin_symmetry:
            # States with undesired spins are shifted up in energy.
            casscf.fix_spin(shift=0.5)
        # supress printing of CASSCF energies
        casscf.verbose = 0
        casscf.run()

        msmd = MultistateMatrixDensityCASSCF(mol, casscf)

        return msmd


class MultistateMatrixDensityTDDFT(MultistateMatrixDensity):
    def __init__(
            self,
            mol,
            rks,
            tddft):
        """
        This class holds the multistate matrix density and can evaluate
        D(r), ∇D(r) and ∇²D(r) on a grid.
        The state densities and transition densities are constructed from
        a TD-DFT calculation with pyscf.

        :param mol: molecule with atomic coordinates, basis set and spin
        :type mol: pyscf.gto.Mole

        :param rks: restricted solution of Kohn-Sham equations with molecular orbitals
        :type rks: pyscf.dft.RKS

        :param tddft: converged solution of the TD-DFT problem
        :type tddf: pyscf.tddft.TDDFT
        """
        # Check input TD-DFT calculation.
        assert numpy.all(tddft.converged), "TD-DFT calculation is not converged."
        assert tddft.singlet == True, "Only singlet excited states are supported."

        # number of atomic orbitals and molecular orbitals
        nao, nmo = rks.mo_coeff.shape

        def density_matrix_mo2ao(dm_mo):
            """
            transform a density matrix in the MO basis in the AO basis

              P^AO_{a,b}   = sum_{m,n} C*_{a,m} P^MO_{m,n} C_{b,n}

            a,b enumerate atomic orbitals, m,n enumerate molecular orbitals
            and C_{a,m} are the self-consistent field MO coefficients.

            :param dm_mo: density matrix in MO basis
            :type dm_mo: numpy.ndarray of shape (nmo,nmo)

            :return dm_ao: density matrix in AO basis
            :rtype dm_ao: numpy.ndarray of shape (nao,nao)
            """
            assert dm_mo.shape == (nmo,nmo)
            dm_ao = numpy.einsum(
                'am,mn,bn->ab',
                rks.mo_coeff, dm_mo, rks.mo_coeff)
            return dm_ao

        # number of electronic states (ground + excited states)
        nstate = tddft.nstates+1
        # Compute the (transition) density matrices in the AO basis.
        nspin = 2

        # A and B matrices from Casida's equation
        #  [A  B] [X]     [0  1] [X]
        #  [    ] [ ] = w [    ] [ ]
        #  [B  A] [Y]     [-1 0] [Y]
        A, B = tddft.get_ab()
        nocc, nvir, _, _ = A.shape
        assert nocc+nvir == nmo

        # Compute (A-B)⁻¹ᐟ²
        AminusB = A-B
        # convert 4D tensor into matrix, (i,a,j,b) -> (ia,jb)
        AminusB = numpy.reshape(AminusB, (nocc*nvir, nocc*nvir))
        # perform linear operations on matrix
        invsqrtAminusB = scipy.linalg.inv(scipy.linalg.sqrtm(AminusB))
        # convert matrix back to 4D tensor (ia,jb) -> (i,a,j,b)
        invsqrtAminusB = numpy.reshape(invsqrtAminusB, (nocc,nvir,nocc,nvir))

        # Although the concept of a wavefunction is alien to density functional theory,
        # the Casida ansatz (Casida 1995), assigns a CIS-like wavefunction to an excited
        # state.
        cis_coefficients = numpy.zeros((nstate-1,nocc,nvir))
        # Loop over excited states and convert excitation (X) and deexcitation (Y)
        # coefficients into coefficients in the basis of singly-excited, spin-adapted
        # configuration functions.
        for istate in range(1, nstate):
            Xi,Yi = tddft.xy[istate-1]
            XplusY = Xi+Yi
            # CIS[istate,o,v] = sqrt(w) (A-B)⁻¹ᐟ² (X+Y)
            cis_coefficients[istate-1,:,:] = (
                numpy.sqrt(tddft.e[istate-1]) *
                numpy.einsum('iajb,jb->ia', invsqrtAminusB, XplusY))
            # Normalize CIS coefficients,
            # this is needed because X and Y are normalized to 1.
            cis_coefficients[istate-1,:,:] /= scipy.linalg.norm(
                cis_coefficients[istate-1,:,:])

            # NOTE: pyscf calculates the transition matrix elements of operators
            #       directly from X+Y (Hermitian operator such as the dipole operator)
            #       or X-Y (non-Hermitian) without an intermediate "wavefunction",
            #       see eqn. (33) in https://doi.org/10.1063/1.4937410 .
            #       Apparently the transition dipoles from the CIS-like "wavefunction"
            #       differ slightly from the correct TD-DFT transition dipoles.
            #
            #       Using the CIS coefficients below leads to the same transition dipoles
            #       that are output by `tddft.transition_dipole()`, but the corresponding
            #       CIS "wavefunction" are not orthonormal.
            #Xi,Yi = tddft.xy[istate-1]
            #cis_coefficients[istate-1,:,:] = 2*(Xi+Yi)

        # Store the CIS coefficients (only needed for unittests)
        self.cis_coefficients = cis_coefficients
        # Store TD-DFT object (only needed for unittests)
        self.tddft = tddft

        # Indices of occupied and virtual orbitals
        occ_indices = numpy.arange(0, nocc)
        vir_indices = numpy.arange(nocc, nocc+nvir)

        # (transition) density matrices
        density_matrices = numpy.zeros((nspin,nstate,nstate,nao,nao))
        for i in range(0, nstate):
            for j in range(0, nstate):
                if i == j:
                    # occupied-occupied, occupied-virtual, virtual-occupied
                    # and virtual-virtual blocks in the (transition) density.
                    oo_block = numpy.ix_(occ_indices, occ_indices)
                    ov_block = numpy.ix_(occ_indices, vir_indices)
                    vo_block = numpy.ix_(vir_indices, occ_indices)
                    vv_block = numpy.ix_(vir_indices, vir_indices)

                    # 1-particle density matrix
                    dm1 = numpy.zeros((nmo,nmo))
                    if i == 0:
                        # of DFT ground state (in MO basis)
                        # Doubly occupied orbitals in the ground state.
                        dm1[occ_indices, occ_indices] = 2.0
                    else:
                        # electron density
                        dm1_electron = numpy.einsum(
                            # sum over occupied orbitals o
                            'ou,ov->uv',
                            cis_coefficients[i-1,:,:],
                            cis_coefficients[i-1,:,:])
                        # hole density
                        dm1_hole = numpy.einsum(
                            # sum over virtual orbitals v
                            'kv,lv->kl',
                            cis_coefficients[i-1,:,:],
                            cis_coefficients[i-1,:,:])
                        # total electron density of excited states
                        #  ρ(i) = ρ0 + ρ(electron) - ρ(hole)
                        # ρ0 - ground state density
                        dm1[occ_indices, occ_indices] = 2.0
                        # ρ(electron)
                        dm1[vv_block] += dm1_electron
                        # ρ(hole)
                        dm1[oo_block] -= dm1_hole

                    dm1_ao = density_matrix_mo2ao(dm1)
                    # for spin-up
                    density_matrices[0,i,i,:,:] = 0.5 * dm1_ao
                    # for spin-down
                    density_matrices[1,i,i,:,:] = 0.5 * dm1_ao
                else:
                    # 1-particle transition density matrix
                    tdm1 = numpy.zeros((nmo,nmo))

                    # between electronic states i and j.
                    if i == 0:
                        # transition density between ground and excited state j
                        tdm1[ov_block] = 0.5 * cis_coefficients[j-1,:,:]
                        tdm1[vo_block] = 0.5 * cis_coefficients[j-1,:,:].transpose()
                    elif j == 0:
                        # transition density between ground and excited state i
                        tdm1[ov_block] = 0.5 * cis_coefficients[i-1,:,:]
                        tdm1[vo_block] = 0.5 * cis_coefficients[i-1,:,:].transpose()
                    else:
                        # transition density between excited states i, j
                        tdm1[vv_block] = numpy.einsum(
                            'ou,ov->uv',
                            cis_coefficients[i-1,:,:],
                            cis_coefficients[j-1,:,:])
                        tdm1[oo_block] = numpy.einsum(
                            'kv,lv->kl',
                            cis_coefficients[i-1,:,:],
                            cis_coefficients[j-1,:,:])
                        for o in occ_indices:
                            tdm1[o,o] -= 2.0 * numpy.einsum(
                                'v,v->',
                                cis_coefficients[i-1,o,:],
                                cis_coefficients[j-1,o,:])

                    tdm1_ao = density_matrix_mo2ao(tdm1)
                    density_matrices[0,i,j,:,:] = 0.5*tdm1_ao
                    # for spin-down
                    density_matrices[1,i,j,:,:] = 0.5*tdm1_ao

        # Total energies of electronic states (electronic energy plus nuclear repulsion)
        eigenenergies = numpy.zeros(nstate)
        # ground state energy
        eigenenergies[0] = rks.e_tot
        # excited state energies
        eigenenergies[1:] = tddft.e_tot

        # Initialize base class.
        super().__init__(mol, eigenenergies, density_matrices)

    @staticmethod
    def create_matrix_density(mol, nstate=4):
        """
        Compute multistate matrix density for the lowest few excited
        singlet states of a small molecule using TD-DFT.

        :param mol: A test molecule with even number of electrons
        :type mol: gto.Mole

        :param nstate: number of electronic states to calculate
        :type nstate: positive int

        :return: multistate matrix density
        :rtype: :class:`~.MultistateMatrixDensity`
        """
        assert nstate > 1, "At least one excited state has to be calculated with TD-DFT"
        rks = pyscf.scf.RKS(mol)
        # supress printing of SCF energy
        rks.verbose = 0
        # compute self-consistent field
        rks.kernel()

        tddft = pyscf.tddft.TDDFT(rks)
        # number of excited states (i.e. excluding the ground state)
        tddft.nstates = nstate-1
        tddft.kernel()

        msmd = MultistateMatrixDensityTDDFT(mol, rks, tddft)

        return msmd
