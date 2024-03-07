#!/usr/bin/env python
# coding: utf-8
import unittest

import numpy
import numpy.testing
import scipy.linalg

from msdft.LinearAlgebra import eigensystem_derivatives
from msdft.LinearAlgebra import LinearAlgebraException


def kinetic_energy_density(L, U, grad_L, grad_U):
    """
    The kinetic energy density matrix

    KEDᵢⱼ(r) = ∑ₐ { 1/8 (∇λₐ·∇λₐ)/λₐ Uᵢₐ Uⱼₐ + 1/2 λₐ ∇Uᵢₐ·∇Uⱼₐ
                      + 1/4 Uᵢₐ (∇λₐ·∇Uⱼₐ) + 1/4 Uⱼₐ (∇λₐ·∇Uᵢₐ) }

    is invariant under reordering of the eigenvalues.

    :param L: eigenvalues λₐ
    :type L: numpy.ndarray of shape (n,)

    :param U: eigenvectors Uᵢₐ
    :type U: numpy.ndarray of shape (n,n)

    :param grad_L: gradients of eigenvalues ∇λₐ
    :type grad_L: numpy.ndarray of shape (n,d)
        where d is the number of partial derivatives
        in the gradient vector.

    :param grad_U: gradients of eigenvectors ∇Uᵢₐ
    :type grad_U: numpy.ndarray of shape (n,n,d)
        where d is the number of partial derivatives
        in the gradient vector.

    :return: kinetic energy density KEDᵢⱼ
    :rtype: numpy.ndarray of shape (n,n)
    """
    # reserve space for kinetic energy density KEDᵢⱼ(r)
    KED = numpy.zeros_like(U)

    # a enumerates non-zero eigenvalues
    #
    # KEDᵢⱼ(r) = ∑ₐ { 1/8 (∇λₐ·∇λₐ)/λₐ Uᵢₐ Uⱼₐ + 1/2 λₐ ∇Uᵢₐ·∇Uⱼₐ
    #                 + 1/4 Uᵢₐ (∇λₐ·∇Uⱼₐ) + 1/4 Uⱼₐ (∇λₐ·∇Uᵢₐ) }

    # compute (∇λ·∇λ)
    grad_L_product = numpy.einsum('ad,ad->a', grad_L, grad_L)
    # compute (∇λ¹ᐟ²·∇λ¹ᐟ²) = 1/4 (∇λ·∇λ)/λ
    grad_sqrtL_product = numpy.zeros_like(L)
    # Avoid dividing by zero for λ=0
    # Non-zero eigenvalues, for which division is not problematic.
    good = abs(L) > 0.0
    grad_sqrtL_product[good] = 1.0/4.0 * grad_L_product[good] / L[good]

    # ∑ₐ 1/2 (∇λₐ¹ᐟ²·∇λₐ¹ᐟ²) Uᵢₐ Uⱼₐ
    KED += 1.0/2.0 * numpy.einsum('a,ia,ja->ij', grad_sqrtL_product, U, U)
    # ∑ₐ 1/2 λₐ ∇Uᵢₐ·∇Uⱼₐ
    KED += 1.0/2.0 * numpy.einsum('a,iad,jad->ij', L, grad_U, grad_U)
    # ∑ₐ 1/4 Uᵢₐ (∇λₐ·∇Uⱼₐ)
    KED += 1.0/4.0 * numpy.einsum('ia,ad,jad->ij', U, grad_L, grad_U)
    # ∑ₐ 1/4 Uⱼₐ (∇λₐ·∇Uᵢₐ)
    KED += 1.0/4.0 * numpy.einsum('ja,ad,iad->ij', U, grad_L, grad_U)

    return KED


class TestLinearAlgebra(unittest.TestCase):
    def check_eigensystem_derivatives(
            self,
            dim=3,
            repeated_eigenvalues=False,
            repeated_eigenvalue_derivatives=False,
            t0=0.234):
        """
        For a one-parameter famility of matrix M(t), the analytical derivatives of
        the eigenvectors and eigenvalues are compared with the numerical ones.
        Complications due to repeated eigenvalues are also checked.

        :param dim: dimension of test matrix
        :type dim: int

        :param repeated_eigenvalues:
            True: The matrix has repeated eigenvalues, but the
            eigenvalue derivatives of the repeated eigenvalues are distinct.
            False: All eigenvalues are distinct.
        :type repeated_eigenvalues: bool

        :param repeated_eigenvalue_derivatives:
            True: The matrix has repeated eigenvalues with repeated eigenvalue
            derivatives.
            False: The derivatives of repeated eigenvalues are dsitinct.
        :type repeated_eigenvalue_derivatives: bool

        :param t0: The parameter at which the derivative M(t)|_{t=t0}
            is taken.
        :type t0: float
        """
        if repeated_eigenvalue_derivatives:
            assert repeated_eigenvalues, (
                "Repeated eigenvalue derivatives are only problematic "
                "if they belong to repeated eigenvalues.")

        # The hardcoded seed ensures that the same random numbers are used
        # every time the test is run. Otherwise the test fails occasionally
        # when the threshold is is too tight.
        random_number_generator = numpy.random.default_rng(seed=6789)

        # In order to verify the code for derivatives of eigenvectors, we have to
        # construct a one-parameter family of symmetric, differentiable matrices S(t),
        # for which the eigenvalue derivatives and eigenvector derivatives are known.
        # We start with differentiable functions for the orthogonal transformation
        # U(t) = exp(f(t) X) and eigenvalues λ1(t), λ2(t), ..., and build the symmetric
        # matrix as
        #   S(t) = U(t).diag(λ1(t), λ2(t), ...).Uᵀ(t)

        # Create a random antisymmetric matrix Xᵀ = -X
        X = random_number_generator.random((dim, dim))
        X = 0.5 * (X - X.T)

        # function f(t) and its derivatives
        def f(t, deriv=0):
            phase = pow(-1, deriv//2)
            if deriv % 2 == 0:
                # f(t), f''(t), .., f^{(2n)}(t)
                return phase * numpy.sin(t)
            else:
                # f'(t), f'''(t), ..., f^{(2n+1)}(t)
                return phase * numpy.cos(t)

        # One-parameter family of orthogonal matrices U(t) = exp(f(t) X)
        # and its derivatives.
        def eigenvectors(t, deriv=0):
            U = scipy.linalg.expm(f(t, deriv=0) * X)
            if deriv == 0:
                return U
            elif deriv == 1:
                return f(t, deriv=1) * numpy.dot(U, X)
            elif deriv == 2:
                UX = numpy.dot(U, X)
                UX2 = numpy.dot(UX, X)
                f1 = f(t, deriv=1)
                f2 = f(t, deriv=2)
                return f2 * UX + pow(f1,2) * UX2
            else:
                raise NotImplementedError("Higher derivatives of orthogonal matrix U(t) not implemented.")

        # A one-parameter family of eigenvalues [λ1(t), λ2(t), ..., λdim(t)]
        def eigenvalues(t, deriv=0):
            if deriv == 0:
                # [λ1(t), λ2(t), ..., λdim(t)]
                evals = numpy.zeros(dim)
                for i in range(0, dim):
                    if repeated_eigenvalues and repeated_eigenvalue_derivatives:
                        evals[i] = pow(t, i//2+1)
                    elif repeated_eigenvalues:
                        evals[i] = pow(t, i//2+1) + i*(t-t0) + i*pow(t-t0,2)
                    else:
                        evals[i] = pow(t, i+1)
                return evals
            elif deriv == 1:
                # first derivatives [λ1'(t), λ2'(t), ..., λdim'(t)]
                evals_deriv = numpy.zeros(dim)
                for i in range(0, dim):
                    if repeated_eigenvalues and repeated_eigenvalue_derivatives:
                        evals_deriv[i] = (i//2+1) * pow(t, i//2)
                    elif repeated_eigenvalues:
                        evals_deriv[i] = (i//2+1) * pow(t, i//2) + i + 2*i*(t-t0)
                    else:
                        evals_deriv[i] = (i+1) * pow(t, i)
                return evals_deriv
            elif deriv == 2:
                # second derivatives [λ1''(t), λ2''(t), ..., λdim''(t)]
                evals_deriv2 = numpy.zeros(dim)
                for i in range(0, dim):
                    if repeated_eigenvalues and repeated_eigenvalue_derivatives:
                        if i//2-1 >= 0:
                            evals_deriv2[i] = (i//2+1) * (i//2) * pow(t, i//2-1)
                    elif repeated_eigenvalues:
                        if i//2-1 >= 0:
                            evals_deriv2[i] = (i//2+1) * (i//2) * pow(t, i//2-1)
                        evals_deriv2[i] += 2*i
                    else:
                        if i-1 >= 0:
                            evals_deriv2[i] = (i+1) * i * pow(t, i-1)
                return evals_deriv2
            else:
                raise NotImplementedError("Higher derivatives are not implemented.")

        # A one-parameter family of symmetric matrices S(t) and its
        # derivatives S'(t) and S''(t).
        def symmetric_matrix(t, deriv=0):
            if deriv == 0:
                # S(t) is assembled from its eigenvalues and eigenvectors,
                #    S(t) = U(t).diag(λ1(t), λ2(t), ...).Uᵀ(t)
                evals = eigenvalues(t)
                U = eigenvectors(t)
                S = numpy.einsum('a,ia,ja->ij', evals, U, U)
                return S
            elif deriv == 1:
                # S'(t) is computed by the chain rule from its eigen decomposition.
                evals = eigenvalues(t, deriv=0)
                evals_deriv = eigenvalues(t, deriv=1)
                U = eigenvectors(t, deriv=0)
                U_deriv = eigenvectors(t, deriv=1)
                S_deriv = (
                    numpy.einsum('a,ia,ja->ij', evals_deriv, U, U) +
                    numpy.einsum('a,ia,ja->ij', evals, U_deriv, U) +
                    numpy.einsum('a,ia,ja->ij', evals, U, U_deriv)
                )
                return S_deriv
            elif deriv == 2:
                # S''(t) is computed by applying the chain rule to its eigen decomposition.
                evals_deriv0 = eigenvalues(t, deriv=0)
                evals_deriv1 = eigenvalues(t, deriv=1)
                evals_deriv2 = eigenvalues(t, deriv=2)
                U_deriv0 = eigenvectors(t, deriv=0)
                U_deriv1 = eigenvectors(t, deriv=1)
                U_deriv2 = eigenvectors(t, deriv=2)
                S_deriv2 = (
                    numpy.einsum('a,ia,ja->ij', evals_deriv2, U_deriv0, U_deriv0) +
                    numpy.einsum('a,ia,ja->ij', evals_deriv1, U_deriv1, U_deriv0) +
                    numpy.einsum('a,ia,ja->ij', evals_deriv1, U_deriv0, U_deriv1) +

                    numpy.einsum('a,ia,ja->ij', evals_deriv1, U_deriv1, U_deriv0) +
                    numpy.einsum('a,ia,ja->ij', evals_deriv0, U_deriv2, U_deriv0) +
                    numpy.einsum('a,ia,ja->ij', evals_deriv0, U_deriv1, U_deriv1) +

                    numpy.einsum('a,ia,ja->ij', evals_deriv1, U_deriv0, U_deriv1) +
                    numpy.einsum('a,ia,ja->ij', evals_deriv0, U_deriv1, U_deriv1) +
                    numpy.einsum('a,ia,ja->ij', evals_deriv0, U_deriv0, U_deriv2)
                )
                return S_deriv2
            else:
                raise NotImplementedError("Higher derivatives are not implemented.")

        t = t0

        # matrix function D(t) and derivatives D'(t), D''(t).
        D = symmetric_matrix(t, deriv=0)
        D_deriv1 = symmetric_matrix(t, deriv=1)
        D_deriv2 = symmetric_matrix(t, deriv=2)

        # Compare analytical derivatives with finite differences.
        dt = 0.0001
        D_plus = symmetric_matrix(t+dt, deriv=0)
        D_deriv1_plus = symmetric_matrix(t+dt, deriv=1)
        D_minus = symmetric_matrix(t-dt, deriv=0)
        D_deriv1_minus = symmetric_matrix(t-dt, deriv=1)
        # finite-difference quotients
        D_deriv1_fd = (D_plus - D_minus)/(2*dt)
        D_deriv2_fd = (D_deriv1_plus - D_deriv1_minus)/(2*dt)

        # Check that D'(t) and D''(t) are implemented correctly
        # by comparing with the finite difference quotiones.
        numpy.testing.assert_almost_equal(D_deriv1_fd, D_deriv1)
        numpy.testing.assert_almost_equal(D_deriv2_fd, D_deriv2)

        # Compute eigenvalues and eigenvector derivatives from D, D' and D''.
        L, U, grad_L, grad_U = eigensystem_derivatives(
            D,
            D_deriv1[:,:,numpy.newaxis],
            D_deriv2[:,:,numpy.newaxis])

        # Reference values for eigenvalues, eigenvectors and their derivatives.
        L_ref = eigenvalues(t, deriv=0)
        U_ref = eigenvectors(t, deriv=0)
        grad_L_ref = eigenvalues(t, deriv=1)[:,numpy.newaxis]
        grad_U_ref = eigenvectors(t, deriv=1)[:,:,numpy.newaxis]

        # Compare the eigenvalues.
        numpy.testing.assert_almost_equal(numpy.sort(L_ref), numpy.sort(L))

        # The eigenvalue/eigenvector derivatives cannot be compared easily if there
        # are repeated eigenvalues, since the order of degenerate eigenvalues
        # is arbitary.
        # The kinetic energy density, on the other hand, is invariant under a reordering
        # of eigenvalues and corresponding eigenvectors.
        KED = kinetic_energy_density(L, U, grad_L, grad_U)
        KED_ref = kinetic_energy_density(L_ref, U_ref, grad_L_ref, grad_U_ref)

        numpy.testing.assert_allclose(KED_ref, KED)

    def test_eigensystem_derivatives_distinct_eigenvalues(self):
        """
        Test eigenvalue derivatives for symmetric matrices
        with distinct eigenvalues.
        """
        # Loop over points t=t0, at which the derivatives of
        # the symmetric matrix S(t) are taken.
        for t0 in [0.01, 0.2345]:
            # matrix size
            for dimension in [2,3,4,5,6,7]:
                with self.subTest(t0=t0, dimension=dimension):
                    self.check_eigensystem_derivatives(
                        dimension,
                        repeated_eigenvalues=False,
                        t0=t0)

    def test_eigensystem_derivatives_repeated_eigenvalues(self):
        """
        Test eigenvalue derivatives for symmetric matrices with
        repeated eigenvalues but distinct eigenvalue derivatives.
        """
        # Loop over points t=t0, at which the derivatives of
        # the symmetric matrix S(t) are taken.
        for t0 in [0.01, 0.2345]:
            # Loop over sizes of matrix
            for dimension in [2,3,4,5,6,7]:
                with self.subTest(t0=t0, dimension=dimension):
                    self.check_eigensystem_derivatives(
                        dimension,
                        repeated_eigenvalues=True,
                        t0=t0)

    def test_raises_exception(self):
        """
        If the repeated eigenvalues have repeated derivatives,
        higher order derivatives of D are needed to determine
        the eigenvector derivatives. This is not implemented and
        an error should be raised if this situation occurs.
        """
        with self.assertRaises(LinearAlgebraException):
            self.check_eigensystem_derivatives(
                4,
                repeated_eigenvalues=True,
                repeated_eigenvalue_derivatives=True,
                t0=0.0001)

if __name__ == "__main__":
    unittest.main()
