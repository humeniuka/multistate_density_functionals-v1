#!/usr/bin/env python
# coding: utf-8
"""
Compare the exact electron repulsion energy matrix between electronic states

  Iᵢⱼ = ∫dx1 ∫dx2...∫dxn Ψ*ᵢ(x1,x2,...,xn) ∑ᵦ<ᵧ 1/|rᵦ-rᵧ| Ψⱼ(x1,x2,...,xn)

with the multi-state local-density approximation

  Iᵢⱼ ≈ Jᵢⱼ[D] - Kᵢⱼ[D] + Cᵢⱼ[D]
      = 1/2 ∑ₖ ∫∫' Dᵢₖ(r) Dₖⱼ(r')/|r-r'|
         - 2¹ᐟ³ Cₓ ∫ [Dᵅ(r)⁴ᐟ³]ᵢⱼ + [Dᵝ(r)⁴ᐟ³]ᵢⱼ dr
         + a ∫ log(Id + b₁ Dᵗᵒᵗ(r)¹ᐟ³ + b₂ Dᵗᵒᵗ(r)²ᐟ³) Dᵗᵒᵗ(r) dr
         + SIC δᵢⱼ

for the water molecule at its equilibrium geometry.
The exact matrix density Dᵢⱼ(r) is calculated using full configuration interaction.
Dᵗᵒᵗ=Dᵅ(r)+Dᵝ(r) is the total density summed over spin.

The self-interaction correction for the core electrons is a constant that neither depends
on the geometry nor on the electronic state.

  SIC = - 2 x [ 1/2 (ρ₁ₛᵅ|ρ₁ₛᵅ) - 2¹ᐟ³ Cₓ ∫ ρ₁ₛᵅ(r)⁴ᐟ³ dr  + ∫ ρ₁ₛᵅ(r) εᶜ(ρ₁ₛᵅ) dr ] δᵢⱼ
"""
import numpy

import pyscf.fci
import pyscf.scf

from msdft.ElectronRepulsionOperators import HartreeLikeFunctional
from msdft.ElectronRepulsionOperators import LDACorrelationLikeFunctional
from msdft.ElectronRepulsionOperators import LSDAExchangeLikeFunctional
from msdft.MultistateMatrixDensity import MultistateMatrixDensityFCI
from msdft.SelfInteractionCorrection import CoreSelfInteractionCorrection


def compare_electron_repulsion_energies(mol, nstate=2):
    """
    The electron repulsion energy matrix is computed exactly
    and approximately using a local functional consisting of
    a Hartree-like and an LSDA-exchange-like term.

    :param nstate: Number of electronic states in the subspace.
       The full CI problem is solved for the lowest nstate states.
    :type nstate: int > 0
    """
    # compute D(r) from full CI
    msmd = MultistateMatrixDensityFCI.create_matrix_density(mol, nstate=nstate)
    # Functionals for parts of electron repulsion.
    # J[D]
    hartree_functional = HartreeLikeFunctional(mol)
    # K[D]
    exchange_functional = LSDAExchangeLikeFunctional(mol)
    # C[D]
    correlation_functional = LDACorrelationLikeFunctional(mol)
    # Remove the self-interaction error (SIE) of the electrons in the core orbitals.
    SIE = CoreSelfInteractionCorrection(mol).total_self_interaction_error()
    # The correction should remove the self-interaction error, therefore the minus sign.
    SIC = -SIE

    # exact electron repulsion
    # Iᵢⱼ = ∫dx1 ∫dx2...∫dxn Ψ*ᵢ(x1,x2,...,xn) ∑ᵦ<ᵧ 1/|rᵦ-rᵧ| Ψⱼ(x1,x2,...,xn)
    I_exact = msmd.exact_electron_repulsion()
    # Jᵢⱼ[D] - Kᵢⱼ[D] + Cᵢⱼ[D] overestimates the electron repulsion because of the
    # self-interaction error. For a single electron, the exchange part should
    # exactly cancel the Hartree part, but this does not happen in the
    # density functional approximation.
    J_Hartree = hartree_functional(msmd)
    K_LSDA = exchange_functional(msmd)
    C_LDA = correlation_functional(msmd)
    # approximate multi-state LSDA electron repulsion
    # Iᵢⱼ ≈ Jᵢⱼ[D] - Kᵢⱼ[D] + Cᵢⱼ[D] + SIC
    I_approximate = J_Hartree - K_LSDA + C_LDA + SIC * numpy.eye(nstate)
    #
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
    print("SIC")
    print(SIC)

    # relative errors
    relative_errors = abs(I_approximate - I_exact)/(abs(I_exact) + 1.0e-8)
    print("=== Relative Errors ===")
    print("|I(approximate)-I(exact)|/|I(exact)|")
    print(relative_errors)


if __name__ == "__main__":
    # water
    mol = pyscf.gto.M(
        atom = 'O  0 0 0; H 0.75 0.00 0.50; H 0.75 0.00 -0.50',
        basis = 'sto-3g',
        charge = 0,
        # singlet
        spin = 0)

    compare_electron_repulsion_energies(mol, nstate=4)
