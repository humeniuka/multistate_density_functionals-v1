#!/usr/bin/env python
# coding: utf-8
"""
Electronic structure of LiF with CASSCF.

References
----------
[1] Werner, Hans‐Joachim, and Wilfried Meyer.
   "MCSCF study of the avoided curve crossing of the two lowest 1Σ+ states of LiF."
    J. Chem. Phys. 74.10 (1981): 5802-5807.
"""
import json
import numpy
import pyscf.fci
import pyscf.gto
import pyscf.mcscf
import pyscf.scf
import pyscf.tools.molden as molden

if __name__ == "__main__":
    # Save exact and approximate electron repulsion energies and other
    # data along the scan.
    scan_data = {
        # bond length (in Å), the scan variable
        'bond_length': [],
        # Total energy (electronic plus nuclear repulsion).
        'eigenenergies': [],
    }

    # Geometry of LiF
    #
    # To fix the signs of the off-diagonal elements of the matrix densities,
    # so that we can plot smooth, continuous curves, the global phases of
    # the wavefunctions have to be aligned with the phases at the previous
    # scan point (the reference).
    msmd_ref = None
    # The experimental LiF bond length is 1.564 Å.
    # The bond length is scanned from 1 to 8 Å.
    bond_lengths = numpy.array(
        [
        1.0,
        # experimental equilibrium geometry
        1.25, 1.564, 1.75,
        2.0, 3.0, 4.0,
        # avoided crossing between the lowest two ¹Σ+ states.
        5.0, 5.5, 6.0, 6.5, 6.6, 6.7, 6.8, 6.9, 7.0, 8.0
        ])
    for bond_length in bond_lengths:
        print(rf"* Li-F bond length : {bond_length:4f} Å")
        mol = pyscf.gto.M(
            atom = f"""
            Li {-bond_length/2} 0 0
            F  { bond_length/2} 0 0
            """,
            # A large basis set is needed.
            basis = 'aug-cc-pvqz',
            charge = 0,
            # singlet
            spin = 0,
            # use symmetry
            symmetry = True)
        print(f"Molecular symmetry is used: {mol.topgroup}")

        hf = pyscf.scf.RHF(mol)
        # supress printing of SCF energy
        hf.verbose = 0
        # compute self-consistent field
        hf.kernel()
        # Show orbital energies, occupancies etc., requires hf.verbose > 0.
        #hf.analyze()
        pyscf.tools.molden.dump_scf(hf, '/tmp/hf.molden', ignore_h=False)

        # CAS: 6 electrons in 21 orbitals
        ncas = 21
        nelecas = 6
        casscf = pyscf.mcscf.CASSCF(hf, ncas, nelecas)

        # In the HF ground state the occupancies for the orbitals by irrep are
        # irrep         A1  E1x  E1y  E2x  E2y
        # electrons      8   2    2    0    0

        # Number of active orbitals in each irrep.
        cas_irrep_nocc = {'A1': 9, 'E1x': 6, 'E1y': 6}
        # Construct the initial guess for the CASSCF orbitals.
        mo_guess = pyscf.mcscf.sort_mo_by_irrep(casscf, hf.mo_coeff, cas_irrep_nocc)

        # Compute lowest 2 singlet states with in the Σ+ irrep.
        nstate = 2
        casscf.nstate = nstate
        # Specify the spatial symmetry of the wavefunction.
        # LiF has C∞v symmetry, the Σ+ irrep is called A1.
        casscf.fcisolver.wfnsym = 'A1'
        # Each state is given equal weight in the CAS energy.
        weights = numpy.array([1.0/nstate] * nstate)
        casscf = casscf.state_average(weights)
        # States with undesired spins are shifted up in energy.
        casscf.fix_spin(shift=0.5)
        casscf.kernel(mo_guess)

        # Output all determinants coefficients.
        for state in range(0, nstate):
            active_orbitals = range(0, ncas)
            occslst = pyscf.fci.cistring.gen_occslst(active_orbitals, nelecas//2)
            print(f'State {state}  Energy= {casscf.e_states[state]}')
            print('   Determinants    CI coefficients')
            for i,occsa in enumerate(occslst):
                for j,occsb in enumerate(occslst):
                    # Only the dominant CI coefficients are printed.
                    ci_coefficient = casscf.ci[state][i,j]
                    if abs(ci_coefficient) < 0.01:
                        continue
                    # The occupation string shows which orbitals are doubly
                    # occupied (2), singly occupied (a or b) or empty (.)
                    # e.g.: '222ab...' for a HOMO-LUMO excited determinant.
                    occupation_string = ''
                    for o in active_orbitals:
                        if o in occsa and o in occsb:
                            # orbital is doubly occupied
                            occupation_string += '2'
                        elif o in occsa:
                            # orbital is singly occupied by spin-up electron
                            occupation_string += 'a'
                        elif o in occsb:
                            # orbital is singly occupied by spin-down electron
                            occupation_string += 'b'
                        else:
                            # orbital is unoccupied
                            occupation_string += '.'
                    print('     %s      %+10.7f' % (occupation_string, ci_coefficient))

        # Save data for later plotting
        scan_data['bond_length'].append(bond_length)
        scan_data['eigenenergies'].append(casscf.e_states.tolist())

        # Save intermediate results, in case the calculation is stopped early.
        with open('eigenenergies_lithium-fluoride.json', 'w') as filehandle:
            json.dump(scan_data, filehandle)

        # Save geometry and orbitals in CAS in molden format.
        with open(f"/tmp/lithium_hydride_{bond_length:3.1f}.molden", "w") as molden_file:
            pyscf.tools.molden.header(mol, molden_file, ignore_h=False)
            pyscf.tools.molden.orbital_coeff(
                mol, molden_file, casscf.mo_coeff, ene=casscf.mo_energy, occ=casscf.mo_occ)
