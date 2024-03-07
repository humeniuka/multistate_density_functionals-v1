#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy
import scipy.linalg


class LinearAlgebraException(Exception):
    pass


def eigensystem_derivatives(D, D_deriv1, D_deriv2=None, epsilon=1.0e-12):
    """
    compute the eigenvalues Λ(t) and eigenvectors U(t) of the
    symmetric matrix D(t), as well as their gradients with
    respect to some external p-dimensional parameter vector t,
    dΛ/dt and dU/dt.

    Repeated eigenvalues are treated correctly (see [1]), however,
    if the derivatives of repeated eigenvalues are also degenerate,
    a :class:`~.LinearAlgebraException` is raised.

    :param D: symmetric matrix
    :type D: numpy.ndarray of shape (n,n)

    :param D_deriv1: D[:,:,p]=dD(t)/dt[p], first derivative of D w/r/t
        the p-th external parameter.
    :type D_deriv1: numpy.ndarray of shape (n,n,p)

    :param D_deriv2: D[:,:,p]=d²D(t)/dt[p]², second derivative of D w/r/t
        the p-th external parameter. When D has repeated eigenvalues,
        D'' is needed to determine the derivatives of the eigenvectors.
        If there are repeated eigenvalues and `D_deriv2` is None, a
        :class:`~.LinearAlgebraException` is raised.
    :type D_deriv2: None or numpy.ndarray of shape (n,n,p)

    :param epsilon: Eigenvalues are considered the same,
        if they differ by less than `epsilon`.
    :type epsilon: float

    :return:
        L, U, L_deriv1, U_deriv1
    :rtype: tuple of numpy.ndarray
        `L` has shape (n), L[j] is the j-th eigenvalue of D.
        `U` has shape (n,n), U[i,j] is the i-th component of the j-th
            eigenvector of D.
        `L_deriv1` has shape (n,p), L_deriv[j,p] is the 1st derivative of the
            j-th eigenvalue of D with respect to the p-th external parameter.
        `U_deriv1` has shape (n,n,p), U_deriv[i,j,p] is the 1st derivative
            of the i-th component of the j-th eigenvector of D
            with respect to the p-th external parameter.

    References
    ----------
    [1] Dailey, R. Lane. "Eigenvector derivatives with repeated eigenvalues."
        AIAA journal 27.4 (1989): 486-491.
    [2] Van Der Aa, N. P. "Computation of eigenvalue and eigenvector derivatives
        for a general complex-valued eigensystem." (2006).
    """
    # Check dimensions of inputs.
    dimension, _, parameters = D_deriv1.shape
    assert D.shape == (dimension, dimension), "Matrix D has to be square."
    assert D_deriv1.shape == (dimension, dimension, parameters)
    if D_deriv2 is not None:
        assert D_deriv2.shape == (dimension, dimension, parameters)

    # Compute eigenvalues Λ and eigenvectors U of the symmetric
    # matrix D.
    L, U = numpy.linalg.eigh(D)

    # Repeated eigenvalues are grouped together.
    # `group_by_eigenvalue[n]` contains a list of indices of all eigenvectors
    # that have eigenvalue `eigenvalue[n]`
    eigenvalues = [L[0]]
    group_by_eigenvalue = [[0]]
    # Eigenvalues in `L` are already sorted, repeated eigenvalues are already
    # grouped together.
    for i in range(1, dimension):
        # Eigenvalues are considered the same if they differ by less than `epsilon`.
        if abs(L[i] - eigenvalues[-1]) < epsilon:
            # The eigenvalue is identical to the previous ones,
            # append it to the last one.
            group_by_eigenvalue[-1].append(i)
        else:
            # Start a new group of eigenvalues.
            group_by_eigenvalue.append([i])
            eigenvalues.append(L[i])

    # If there are repeated eigenvalues, any linear combination of degenerate eigenvectors
    # is an eigenvector, however, we have to find the rotation for which the eigenvectors
    # change smoothly and are thus differentiable.

    # The same rotation matrix Γ₂ should result for all partial derivatives,
    # otherwise the matrix function is not differentiable.
    for p in range(0, parameters):
        for eigenvalue, group in zip(eigenvalues, group_by_eigenvalue):
            degeneracy = len(group)
            if degeneracy > 1:
                if D_deriv2 is None:
                    raise LinearAlgebraException(
                        "Matrix D has repeated eigenvalues. The second derivative D'' "
                        "has to be provided to determine the eigenvector derivatives.")
                # There are repeated eigenvalues, but we assume that the
                # derivatives of the repeated eigenvalues are all distinct.

                # This corresponds to section 3.2 in
                # N.P. van der Aa 2006,
                # "Computation of eigenvalue and eigenvector derivatives
                #  for a general complex-valued eigensystem",

                # The columns of X2 are the eigenvectors belonging to the
                # repeated eigenvalues.
                X2 = U[:,group]
                # Since D is a symmetric matrix, the eigenvectors are orthogonal,
                # so that Y2 = X2^-1 = X2^t
                Y2 = X2.T
                # Form Y2.D'.X2
                UtD1U_degenerate = numpy.dot(Y2, numpy.dot(D_deriv1[:,:,p], X2))
                # Solve the eigenvalue problem
                #   (Uᵀ.D'.U).Γ₂ = Γ₂.Λ'
                L_deriv1_test, Gamma2 = scipy.linalg.eigh(UtD1U_degenerate)
                # The degenerate eigenvectors are rotated by the orthogonal transformation Γ₂.
                U[:,group] = numpy.dot(U[:,group], Gamma2)

    # Compute the eigenvalue derivatives.
    #
    # If uᵢ is a normalized eigenvector of D with eigenvalue λᵢ,
    #   D uᵢ = λᵢ uᵢ
    # then the 1st derivative of the eigenvalue is given by
    #   λᵢ' = <uᵢ,D' uᵢ>
    # where < , > is the scalar product.
    #
    # Λ'(r), 1st derivatives of eigenvalues of D(r)
    L_deriv1 = numpy.einsum('ki,klp,li->ip', U, D_deriv1, U)

    # Uᵀ.D'.U
    UtD1U = numpy.einsum('ki,klp,lj->ijp', U, D_deriv1, U)

    if D_deriv2 is not None:
        # Uᵀ.D''.U
        UtD2U = numpy.einsum('ki,klp,lj->ijp', U, D_deriv2, U)

    # Compute Cᵢⱼ = 1/(λᵢ-λⱼ) ∑ₖ,ₗ Uₖᵢ D'ₖₗ Uₗⱼ   for i ≠ j and λᵢ ≠ λⱼ.
    # Since eigenvectors are normalized, C is an antisymmetric matrix,
    #   Cᵀ=-C,
    # its diagonal elements are zero. For degenerate eigenvalues, λᵢ = λⱼ,
    # the elements of C are determined later.
    C = numpy.zeros((dimension, dimension, parameters))
    for i in range(0, dimension):
        for j in range(0, dimension):
            if i == j or abs(L[i] - L[j]) < epsilon:
                # skip diagonal terms and blocks for degenerate eigenvalues
                continue
            C[i,j,:] = UtD1U[i,j,:]/(L[j]-L[i])

    # The derivatives w/r/t each external parameter can be treated separately.
    for p in range(0, parameters):
        for eigenvalue, group in zip(eigenvalues, group_by_eigenvalue):
            degeneracy = len(group)
            if degeneracy == 1:
                # Nothing to do, C is already fully determined.
                pass
            else:
                # Check that there are no repeated eigenvalue derivatives within one group
                # of repeated eigenvalues, otherwise we would need the 3rd derivative of D
                # to determine the eigenvector derivatives.
                eigenvalue_deriv1_sorted = numpy.sort(L_deriv1[group,p])
                # Adjacent eigenvalues in the sorted array have to differ by more than EPSILON
                difference = eigenvalue_deriv1_sorted[1:] - eigenvalue_deriv1_sorted[:-1]
                if not numpy.all(difference > epsilon):
                    # If the repeated eigenvalues belonging to the repeated
                    # eigenvalue derivatives are zero, we don't care about the
                    # correct eigenvector derivatives, since zero eigenvalues do
                    # not contribute to the kinetic energy density.
                    if numpy.all(abs(L[group]) <= epsilon):
                        # Zero eigenvalues can be ignored.
                        continue
                    # Repeated eigenvalues and eigenvalue derivatives cannot be treated correctly
                    # by this implementation.
                    raise LinearAlgebraException(
                        "Derivatives of repeated eigenvalues have to be distinct.")

                # Loop over degenerate eigenvectors associated with the same eigenvalue.
                for iG in group:
                    for jG in group:
                        if iG == jG:
                            # C is an antisymmetric matrix, so C[iG,iG] = 0
                            continue
                        C[iG,jG,p] = -0.5 * UtD2U[iG,jG,p]
                        # Loop over other eigenvalues which are not identical to L[iG].
                        for lO in range(0, dimension):
                            if lO not in group:
                                C[iG,jG,p] -= (L[lO] - eigenvalue) * C[iG,lO,p] * C[lO,jG,p]

                        C[iG,jG,p] /= (L_deriv1[iG,p] - L_deriv1[jG,p])

    # Eigenvector derivatives
    # U' = U.C
    U_deriv1 = numpy.einsum('ik,kjp->ijp', U, C)

    return L, U, L_deriv1, U_deriv1
