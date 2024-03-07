#!/usr/bin/env python
# coding: utf-8
import numpy
import scipy.linalg

class BasisTransformation(object):
    def __init__(self, L):
        self.L = L

    def transform_vector(self, fcivecs):
        """
        rotate CI vectors by the basis transformation

          v' = L v

        The transformed electronic states are linear combinations of
        the old ones.

        :param fcivecs: list of CI vectors
        :type fcivecs: list of numpy.ndarray

        :return fcivecs_transformed: transformed CI vectors
        :rtype fcivecs_transformed: list of numpy.ndarray
        """
        # Number of electronic states
        nstate = len(fcivecs)
        # Convert list into an array of shape (nstate,...,...)
        v = numpy.asarray(fcivecs)
        v_transformed = numpy.einsum('ab,buv->auv', self.L, v)
        # Convert the transformed array back into a list of CI vectors
        fcivecs_transformed = [v_transformed[i] for i in range(0, nstate)]

        return fcivecs_transformed

    def transform_operator(self, matrix):
        """
        perform a similarity transform on an operator O

          O' = L O Lᵗ

        :param matrix: matrix representation of the operator O
          in the subspace
        :type matrix: numpy.ndarray of shape (n,n)

        :return transformed_matrix: transformed operator O'
          in the subspace
        :type matrix: numpy.ndarray of shape (n,n)
        """
        # check dimensions are compatible
        assert matrix.shape == self.L.shape

        transformed_matrix = self.L @ matrix @ self.L.conjugate().transpose()
        return transformed_matrix
        
    @classmethod
    def random(self, dimension, scale=1.0):
        """
        generate a random, orthogonal matrix.

          L = exp(-κ)

        where κ is a real, antisymmetric matrix

          κᵗ = -κ

        so that LᵗL = exp(κ-κ) = 1

        :param dimension: dimension n
        :type dimension: int
        
        :param scale: The random numbers in κ are chosen from the
          range [-scale,scale]
        :type scale: float

        :return L: orthogonal n x n matrix
        :rtype L: numpy.ndarray of shape (n,n)
        """
        # random element of the Lie algebra
        n = dimension
        kappa = 2.0 * scale * (numpy.random.rand(n,n) - 0.5)
        
        # Make kappa an antisymmetric matrix
        kappa = 0.5 * (kappa - kappa.transpose())
        
        # The matrix exponential gives the orthogonal matrix
        L = scipy.linalg.expm(kappa)

        return BasisTransformation(L)


