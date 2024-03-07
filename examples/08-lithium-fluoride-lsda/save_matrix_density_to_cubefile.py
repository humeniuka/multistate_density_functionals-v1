#!/usr/bin/env python
# coding: utf-8
"""
Save the CASSCF matrix density to a series of cube files.
"""
import numpy
import pyscf.tools.cubegen
import pyscf.tools.molden

from msdft.MultistateMatrixDensity import MultistateMatrixDensityCASSCF


if __name__ == "__main__":
    # Lithium-fluoride.

    # The experimental LiF bond length is 1.564 Å.
    # The avoided crossing between the two Σ+ states lies approximately at 6.8 Å.
    #bond_length = 1.564
    bond_length = 6.8

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
    print("CASSCF calculation ...")
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

    # Compute D(r) from CASSCF.
    msmd = MultistateMatrixDensityCASSCF(mol, casscf)

    # Write the (transition) densities D(r) to cube files.
    cube = pyscf.tools.cubegen.Cube(mol)
    # Evaluate D(r) on the rectangular grid of the cube.
    coords = cube.get_coords()
    D, _, _ = msmd.evaluate(coords)

    # Loop over electronic states.
    for istate in range(0, nstate):
        for jstate in range(istate, nstate):
            if istate == jstate:
                comment = f"Density of state {istate}"
            else:
                comment = f"Transition density between states {istate} and {jstate}"
            # spin up Dᵢⱼ(r)
            field = numpy.reshape(D[0,istate,jstate,:], (cube.nx, cube.ny, cube.nz))
            cube.write(
                field,
                f'/tmp/matrix_density_{istate}_{jstate}_up.cube',
                comment=comment + " (spin up)")
            # spin down Dᵢⱼ(r)
            field = numpy.reshape(D[1,istate,jstate,:], (cube.nx, cube.ny, cube.nz))
            cube.write(
                field,
                f'/tmp/matrix_density_{istate}_{jstate}_down.cube',
                comment=comment + " (spin down)")
            # Total spin density
            field = numpy.reshape(
                D[0,istate,jstate,:]+D[1,istate,jstate,:], (cube.nx, cube.ny, cube.nz))
            cube.write(
                field,
                f'/tmp/matrix_density_{istate}_{jstate}.cube',
                comment=comment + " (spin up + spin down)")

    print("Cube files were saved to /tmp/matrix_density_[istate]_[jstate]_[spin].cube")
