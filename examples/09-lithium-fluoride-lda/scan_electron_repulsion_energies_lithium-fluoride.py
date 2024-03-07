#!/usr/bin/env python
# coding: utf-8
"""
compare the exact electron repulsion energy matrix between electronic states

    Wᵢⱼ = ∫dx1 ∫dx2...∫dxn Ψ*ᵢ(x1,x2,...,xn) ∑ᵦ<ᵧ 1/|rᵦ-rᵧ| Ψⱼ(x1,x2,...,xn)

with the multi-state local-density approximation

    Wᵢⱼ ≈ Jᵢⱼ[D] - Kᵢⱼ[D] + Cᵢⱼ[D] + SIC δᵢⱼ
        = 1/2 ∑ₖ ∫∫' Dᵢₖ(r) Dₖⱼ(r')/|r-r'|
          - Cₓ ∫ [D(r)⁴ᐟ³]ᵢⱼ dr
          + a ∫ log(Id + b₁ D(r)¹ᐟ³ + b₂ D(r)²ᐟ³ ) D(r) dr
          + SIC δᵢⱼ

for a range of LiF bond lengths.
The matrix density Dᵢⱼ(r) is calculated using CASSCF.

The self-interaction correction for the core electrons is a constant that neither depends
on the geometry nor on the electronic state.

    SIC = - 2 x [ 1/2 (ρ₁ₛᵅ|ρ₁ₛᵅ) - 2¹ᐟ³ Cₓ ∫ ρ₁ₛᵅ(r)⁴ᐟ³ dr  + ∫ ρ₁ₛᵅ(r) εᶜ(ρ₁ₛᵅ) dr ] δᵢⱼ

"""
import json
import numpy

import pyscf.fci
import pyscf.gto
import pyscf.mcscf
import pyscf.scf
import pyscf.tools.molden as molden

from msdft.ElectronRepulsionOperators import HartreeLikeFunctional
from msdft.ElectronRepulsionOperators import LDACorrelationLikeFunctional
from msdft.ElectronRepulsionOperators import LDAExchangeLikeFunctional
from msdft.KineticOperatorFunctional import LDAThomasFermiFunctional
from msdft.KineticOperatorFunctional import LDAVonWeizsaeckerFunctional
from msdft.MultistateMatrixDensity import MultistateMatrixDensityCASSCF
from msdft.SelfInteractionCorrection import CoreSelfInteractionCorrection


if __name__ == "__main__":
    print("This calculation can take a while. Please be patient ...")
    # Save exact and approximate electron repulsion energies and other
    # data along the scan.
    scan_data = {
        # The bond length (in Å), the scan variable
        'bond_length': [],
        # Total energy (electronic plus nuclear repulsion).
        'eigenenergies': [],
        # electron repulsion matrix
        'W_exact': [],
        # approximate electron repulsion matrix and its constituents.
        'W_approximate': [],
        'J_Hartree': [],
        'K_LDA': [],
        'C_LDA': [],
        # The self-interaction correction for the core orbitals, a constant
        # that does not depend on the geometry and is the same for all
        # electronic states.
        'self_interaction_correction': [],
        # kinetic energy
        'T_exact': [],
        # approximate kinetic energy
        'T_approximate': [],
        'T_ThomasFermi': [],
        'T_vonWeizsaecker': [],
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

        # Construct matrix density D(r) from the CASSCF wavefunctions.
        msmd = MultistateMatrixDensityCASSCF(mol, casscf)

        # Remove the self-interaction error (SIE) of the electrons in the core orbitals.
        SIE = CoreSelfInteractionCorrection(mol).total_self_interaction_error()
        # The correction should remove the self-interaction error, therefore the minus sign.
        SIC = -SIE

        # Align the signs of the wavefunctions with those at the privious scan point.
        # At the first scan point there is no reference.
        if msmd_ref is not None:
            msmd.align_phases(msmd_ref)
        # Functionals for parts of electron repulsion.
        # J[D]
        hartree_functional = HartreeLikeFunctional(mol)
        # K[D]
        exchange_functional = LDAExchangeLikeFunctional(mol)
        # C[D]
        correlation_functional = LDACorrelationLikeFunctional(mol)
        # kinetic energy
        # T[D] = T_{TF}[D] + 1/9 T_{vW}[D]
        kinetic_functional_TF = LDAThomasFermiFunctional(mol)
        kinetic_functional_vW = LDAVonWeizsaeckerFunctional(mol)

        # exact electron repulsion
        # Wᵢⱼ = ∫dx1 ∫dx2...∫dxn Ψ*ᵢ(x1,x2,...,xn) ∑ᵦ<ᵧ 1/|rᵦ-rᵧ| Ψⱼ(x1,x2,...,xn)
        W_exact = msmd.exact_electron_repulsion()
        # approximate multi-state LDA electron repulsion
        # Wᵢⱼ ≈ Jᵢⱼ[D] - Kᵢⱼ[D] + Cᵢⱼ[D] + SIC δᵢⱼ
        print("calculating J[D] ...")
        J_Hartree = hartree_functional(msmd)
        print("calculating K[D] ...")
        K_LDA = exchange_functional(msmd)
        print("calculating C[D] ...")
        C_LDA = correlation_functional(msmd)
        W_approximate = J_Hartree - K_LDA + C_LDA + SIC * numpy.eye(nstate)

        # Compare approximate and exact electron repulsion.
        print("=== Electron Repulsion Matrices ===")
        print("W_exact")
        print(W_exact)
        print("W_approximate")
        print(W_approximate)
        print("J Hartree")
        print(J_Hartree)
        print("-K_LDA")
        print(-K_LDA)
        print("C_LDA")
        print(C_LDA)
        print("self-interaction correction for core orbitals")
        print(SIC)

        # relative errors
        relative_errors = abs(W_approximate - W_exact)/(abs(W_exact) + 1.0e-8)
        print("=== Relative Errors ===")
        print("|W(approximate)-W(exact)|/|W(exact)|")
        print(relative_errors)

        # Orbital-free kinetic functionals.
        print("calculating T_FH[D] ...")
        T_TF = kinetic_functional_TF(msmd)
        print("calculating T_vW[D] ...")
        T_vW = kinetic_functional_vW(msmd)
        # The factor 1/9 is explained in section 6.7 "Conventional gradient correction"
        # in Parr & Yang's book.
        T_approximate = T_TF + 1.0/9.0 * T_vW

        # Compare approximate and exact kinetic energy.
        T_exact = msmd.exact_kinetic_energy()
        print("=== Kinetic Energy Matrices ===")
        print("T_exact")
        print(T_exact)
        print("T_approximate")
        print(T_approximate)
        print("T_TF")
        print(T_TF)
        print("T_vW")
        print(T_vW)

        # relative errors
        relative_errors = abs(T_approximate - T_exact)/(abs(T_exact) + 1.0e-8)
        print("=== Relative Errors ===")
        print("|T(approximate)-T(exact)|/|T(exact)|")
        print(relative_errors)

        # Save data for later plotting
        scan_data['bond_length'].append(bond_length)
        scan_data['eigenenergies'].append(casscf.e_states.tolist())
        scan_data['W_exact'].append(W_exact.tolist())
        scan_data['W_approximate'].append(W_approximate.tolist())
        scan_data['J_Hartree'].append(J_Hartree.tolist())
        scan_data['K_LDA'].append(K_LDA.tolist())
        scan_data['C_LDA'].append(C_LDA.tolist())
        scan_data['self_interaction_correction'].append(SIC)
        scan_data['T_exact'].append(T_exact.tolist())
        scan_data['T_approximate'].append(T_approximate.tolist())
        scan_data['T_ThomasFermi'].append(T_TF.tolist())
        scan_data['T_vonWeizsaecker'].append(T_vW.tolist())

        # This scan point becomes the reference for the next scan point.
        msmd_ref = msmd

        # Save intermediate results, in case the calculation is stopped early.
        with open('electron_repulsion_energies_lithium-fluoride.json', 'w') as filehandle:
            json.dump(scan_data, filehandle)
