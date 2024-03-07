#!/usr/bin/env python
# coding: utf-8
import unittest

import numpy
import numpy.testing

from msdft.BasisTransformation import BasisTransformation

class TestBasisTransformation(unittest.TestCase):
    def test_random_orthogonal_matrix(self):
        """
        Check that the random orthogonal matrix fulfills

          LᵗL = 1
        """
        dimension = 4
        Id = numpy.eye(dimension)
        basis_transformation = BasisTransformation.random(dimension)
        L = basis_transformation.L
        LtL = L.transpose() @ L

        numpy.testing.assert_almost_equal(Id, LtL)

    def test_transform_vector(self):
        """
        Check that the orthogonal transformation preserves the scalar product

          <Lv,Lv> = vᵗLᵗLv = <v,v>
        """
        dimension = 5
        basis_transformation = BasisTransformation.random(dimension)
        # random complex vector
        v = 3.6346 * 2.0 * (numpy.random.rand(dimension, 6,3) - 0.5)
        Lv = basis_transformation.transform_vector(v)
        Lv = numpy.asarray(Lv)
        # scalar product of the original vector, <v,v>
        v2 = numpy.einsum('iab,iab->', v, v)
        # scalar product of the transformed vector, <Lv,Lv>
        vtLtLv = numpy.einsum('iab,iab->', Lv, Lv)

        numpy.testing.assert_almost_equal(vtLtLv, v2)

    def test_transform_operator(self):
        """
        Check that the trace of a matrix is not changed by a similarity transformation

          tr(O') = tr(L O Lᵗ) = tr(O Lᵗ L)= tr(O)
        """
        dimension = 5
        basis_transformation = BasisTransformation.random(dimension)
        # random complex matrix O and tr(O)
        matrix = 1.2435 * 2.0 * (numpy.random.rand(dimension, dimension) - 0.5)
        trace = numpy.trace(matrix)
        # compute O' and tr(O')
        transformed_matrix = basis_transformation.transform_operator(matrix)
        transformed_trace = numpy.trace(transformed_matrix)

        # check that tr(O) == tr(O')
        self.assertAlmostEqual(trace, transformed_trace)
        

if __name__ == "__main__":
    unittest.main()

