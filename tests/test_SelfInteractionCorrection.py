#!/usr/bin/env python
# coding: utf-8
import numpy
import numpy.testing

import pyscf.gto

import unittest

from msdft.SelfInteractionCorrection import CoreSelfInteractionCorrection


class TestCoreSelfInteractionCorrection(unittest.TestCase):
    def create_test_molecules(self):
        """ dictionary with different molecules to run the tests on """
        molecules = {
            # 1-electron systems
            'hydrogen atom': pyscf.gto.M(
                atom = 'H 0 0 0',
                basis = 'sto-3g',
                # doublet
                spin = 1),
            # many electrons
            'oxygen atom': pyscf.gto.M(
                atom = 'O 0 0 0',
                basis = 'sto-3g',
                # singlet
                spin = 0),
            'oxygen molecule': pyscf.gto.M(
                atom = 'O 0.0 0.0 0.0;  O 0.0 0.0 1.21',
                basis = 'sto-3g',
                # singlet
                spin = 0),
            'water': pyscf.gto.M(
                atom = 'O  0 0 0; H 0.75 0.00 0.50; H 0.75 0.00 -0.50',
                basis = 'sto-3g',
                # singlet
                spin = 0),
        }
        return molecules

    def test_total_self_interaction_error(self):
        """
        Check that the self-interaction of the cores of different atoms are added correctly.
        """
        molecules = self.create_test_molecules()
        SIE_water = CoreSelfInteractionCorrection(
            molecules['water']).total_self_interaction_error()
        SIE_oxygen_molecule = CoreSelfInteractionCorrection(
            molecules['oxygen molecule']).total_self_interaction_error()
        SIE_oxygen_atom = CoreSelfInteractionCorrection(
            molecules['oxygen atom']).total_self_interaction_error()
        SIE_hydrogen_atom = CoreSelfInteractionCorrection(
            molecules['hydrogen atom']).total_self_interaction_error()

        # Hydrogen has no core electrons
        self.assertAlmostEqual(SIE_hydrogen_atom, 0.0)
        # SIE is determined by element type, the geometry does not matter.
        self.assertAlmostEqual(2*SIE_oxygen_atom, SIE_oxygen_molecule)
        self.assertAlmostEqual(SIE_oxygen_atom, SIE_water)

        # Check the actual value (in Hartree)
        self.assertAlmostEqual(0.5013, SIE_oxygen_atom, places=3)

    def test_self_interaction_energy_of_core(self):
        """
        Test the static function
        `CoreSelfInteractionCorrection.self_interaction_energy_of_core`
        """
        molecules = self.create_test_molecules()
        SIE_oxygen_atom = CoreSelfInteractionCorrection(
            molecules['oxygen atom']).total_self_interaction_error()

        SIEs_core_orbitals = CoreSelfInteractionCorrection.self_interaction_energy_of_core(
            'O', 'sto-3g')
        # Oxygen has one 1s core orbital
        self.assertEqual(1, len(SIEs_core_orbitals))
        self.assertAlmostEqual(SIE_oxygen_atom, numpy.sum(SIEs_core_orbitals))

        # Check heavy atoms with multipl core orbitals.
        # In Silicon the 1s,2s,2px,2py,2pz orbitals are part of the core
        SIEs_core_orbitals = CoreSelfInteractionCorrection.self_interaction_energy_of_core(
            'Si', 'sto-3g')
        self.assertEqual(5, len(SIEs_core_orbitals))


if __name__ == "__main__":
    unittest.main(failfast=True)
