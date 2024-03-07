#!/usr/bin/env python
# coding: utf-8
"""
compare the exact electron repulsion energy matrix between electronic states

    Iᵢⱼ = ∫dx1 ∫dx2...∫dxn Ψ*ᵢ(x1,x2,...,xn) ∑ᵦ<ᵧ 1/|rᵦ-rᵧ| Ψⱼ(x1,x2,...,xn)

with the multi-state local-density approximation

    Iᵢⱼ ≈ Jᵢⱼ[D] - Kᵢⱼ[D] + Cᵢⱼ[D] + SIC δᵢⱼ
        = 1/2 ∑ₖ ∫∫' Dᵢₖ(r) Dₖⱼ(r')/|r-r'|
          - 2¹ᐟ³ Cₓ ∫ [Dᵅ(r)⁴ᐟ³]ᵢⱼ + [Dᵝ(r)⁴ᐟ³]ᵢⱼ dr
          + a ∫ log(Id + b₁ D(r)¹ᐟ³ + b₂ D(r)²ᐟ³ ) D(r) dr
          + SIC δᵢⱼ

for a range of the ethylene geometries with different torsion angles around the C=C
double bond.
The matrix density Dᵢⱼ(r) is calculated using CASSCF.

The self-interaction correction for the core electrons is a constant that neither depends
on the geometry nor on the electronic state.

    SIC = - 2 x [ 1/2 (ρ₁ₛᵅ|ρ₁ₛᵅ) - 2¹ᐟ³ Cₓ ∫ ρ₁ₛᵅ(r)⁴ᐟ³ dr  + ∫ ρ₁ₛᵅ(r) εᶜ(ρ₁ₛᵅ) dr ] δᵢⱼ

"""
import json
import numpy

import pyscf.fci
import pyscf.scf

from msdft.ElectronRepulsionOperators import HartreeLikeFunctional
from msdft.ElectronRepulsionOperators import LDACorrelationLikeFunctional
from msdft.ElectronRepulsionOperators import LSDAExchangeLikeFunctional
from msdft.MultistateMatrixDensity import MultistateMatrixDensityCASSCF
from msdft.SelfInteractionCorrection import CoreSelfInteractionCorrection


if __name__ == "__main__":
    print("This calculation can take a while. Please be patient ...")
    # Save exact and approximate electron repulsion energies and other
    # data along the scan.
    scan_data = {
        # torsion around carbon-carbon double bond (in °), the scan variable
        'torsion_angle': [],
        # Total energy (electronic plus nuclear repulsion).
        'eigenenergies': [],
        # electron repulsion matrix
        'I_exact': [],
        # approximate electron repulsion matrix and its constituents.
        'I_approximate': [],
        'J_Hartree': [],
        'K_LSDA': [],
        'C_LDA': [],
        # The self-interaction correction for the core orbitals, a constant
        # that does not depend on the geometry and is the same for all
        # electronic states.
        'self_interaction_correction': [],
    }

    # Geometry of ethene
    #
    # To fix the signs of the off-diagonal elements of the matrix densities,
    # so that we can plot smooth, continuous curves, the global phases of
    # the wavefunctions have to be aligned with the phases at the previous
    # scan point (the reference).
    msmd_ref = None
    # The experimental HCH angle is 117.6°.
    angleHCH = 117.6 * numpy.pi/180.0
    # The experimental CH bond length is 1.086 Å.
    rCH = 1.086
    # The experimental carbon-carbon bond length of ethene (NIST) is 1.339 Å.
    rCC = 1.339
    # The torsion angle is scanned from 0° to 90°.
    torsion_angles = numpy.linspace(0.0, 90.0, 10) * numpy.pi/180.0
    for torsion_angle in torsion_angles:
        print(rf"* H2C=CH2 torsion angle : {torsion_angle*180.0/numpy.pi:4f}°")
        # The hydrogen atoms of the fixed H2C= group lie in the xy plane.
        xH = rCC/2 + rCH * numpy.cos(angleHCH/2.0)
        yH = rCH * numpy.sin(angleHCH/2.0)
        # The C=C bond is parallel to the x-axis.
        # The mobile =CH2 group is rotated around the x-axis.
        mol = pyscf.gto.M(
            atom = f"""
            C {-rCC/2} 0 0
            C { rCC/2} 0 0
            H { xH} { numpy.cos(torsion_angle) * yH} { numpy.sin(torsion_angle) * yH}
            H { xH} {-numpy.cos(torsion_angle) * yH} {-numpy.sin(torsion_angle) * yH}
            H {-xH} { yH} 0
            H {-xH} {-yH} 0
            """,
            # small basis set
            basis = '6-31g*',
            charge = 0,
            # singlet
            spin = 0)

        # Remove the self-interaction error (SIE) of the electrons in the core orbitals.
        SIE = CoreSelfInteractionCorrection(mol).total_self_interaction_error()
        # The correction should remove the self-interaction error, therefore the minus sign.
        SIC = -SIE

        # Compute D(r) from CASSCF for the lowest 3 singlet states.
        nstate = 3

        msmd = MultistateMatrixDensityCASSCF.create_matrix_density(
          mol,
          nstate=nstate,
          # The complete active space consists of 2 electrons in 2 orbitals
          ncas=2,
          nelecas=2
        )
        # Align the signs of the wavefunctions with those at the privious scan point.
        # At the first scan point there is no reference.
        if msmd_ref is not None:
            msmd.align_phases(msmd_ref)
        # Functionals for parts of electron repulsion.
        # J[D]
        hartree_functional = HartreeLikeFunctional(mol)
        # K[D]
        exchange_functional = LSDAExchangeLikeFunctional(mol)
        # C[D]
        correlation_functional = LDACorrelationLikeFunctional(mol)

        # exact electron repulsion
        # Iᵢⱼ = ∫dx1 ∫dx2...∫dxn Ψ*ᵢ(x1,x2,...,xn) ∑ᵦ<ᵧ 1/|rᵦ-rᵧ| Ψⱼ(x1,x2,...,xn)
        I_exact = msmd.exact_electron_repulsion()
        # approximate multi-state LSDA electron repulsion
        # Iᵢⱼ ≈ Jᵢⱼ[D] - Kᵢⱼ[D] + Cᵢⱼ[D] + SIC δᵢⱼ
        print("calculating J[D] ...")
        J_Hartree = hartree_functional(msmd)
        print("calculating K[D] ...")
        K_LSDA = exchange_functional(msmd)
        print("calculating C[D] ...")
        C_LDA = correlation_functional(msmd)
        I_approximate = J_Hartree - K_LSDA + C_LDA + SIC * numpy.eye(nstate)

        # Compare approximate and exact electron repulsion.
        print("=== Electron Repulsion Matrices ===")
        print("I_exact")
        print(I_exact)
        print("I_approximate")
        print(I_approximate)
        print("J Hartree")
        print(J_Hartree)
        print("-K_LSDA")
        print(-K_LSDA)
        print("C_LDA")
        print(C_LDA)
        print("self-interaction correction for core orbitals")
        print(SIC)

        # relative errors
        relative_errors = abs(I_approximate - I_exact)/(abs(I_exact) + 1.0e-8)
        print("=== Relative Errors ===")
        print("|C(approximate)-C(exact)|/|C(exact)|")
        print(relative_errors)

        # Save data for later plotting
        scan_data['torsion_angle'].append(torsion_angle*180.0/numpy.pi)
        scan_data['eigenenergies'].append(msmd.eigenenergies.tolist())
        scan_data['I_exact'].append(I_exact.tolist())
        scan_data['I_approximate'].append(I_approximate.tolist())
        scan_data['J_Hartree'].append(J_Hartree.tolist())
        scan_data['K_LSDA'].append(K_LSDA.tolist())
        scan_data['C_LDA'].append(C_LDA.tolist())
        scan_data['self_interaction_correction'].append(SIC)

        # This scan point becomes the reference for the next scan point.
        msmd_ref = msmd

        # Save intermediate results, in case the calculation is stopped early.
        with open('electron_repulsion_energies_ethene_cc_torsion.json', 'w') as filehandle:
            json.dump(scan_data, filehandle)
