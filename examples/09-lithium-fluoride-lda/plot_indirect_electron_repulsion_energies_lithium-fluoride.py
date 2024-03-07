#!/usr/bin/env python
# coding: utf-8
"""
plot the indirect part of the exact matrix elements of the electron repulsion operator

  Wᵢⱼ = ∫dx1 ∫dx2...∫dxn Ψ*ᵢ(x1,x2,...,xn) ∑ᵦ<ᵧ 1/|rᵦ-rᵧ| Ψⱼ(x1,x2,...,xn)

i.e.

  Wᵢⱼ - Jᵢⱼ[D]

and compare it with the multistate Thomas-Fermi-Dirac approximation for
exchange (X=-K) and correlation (C),

  -Kᵢⱼ[D] + Cᵢⱼ[D] + SIC δᵢⱼ,

for the diagonal (i=j) and off-diagonal (i≠j) elements for all
scan geometries.
"""
import json
import matplotlib.lines
import matplotlib.pyplot as plt

import numpy

if __name__ == "__main__":
    plt.style.use('./latex.mplstyle')
    # Load scan data
    with open('electron_repulsion_energies_lithium-fluoride.json', 'r') as filehandle:
        scan_data = json.load(filehandle)

    # Bond lengths are in Angstrom.
    bond_length = numpy.array(scan_data['bond_length'])
    # All energies are in Hartree.
    eigenenergies = numpy.array(scan_data['eigenenergies'])
    W_exact = numpy.array(scan_data['W_exact'])
    W_approximate = numpy.array(scan_data['W_approximate'])
    J_Hartree = numpy.array(scan_data['J_Hartree'])
    K_LDA = numpy.array(scan_data['K_LDA'])
    C_LDA = numpy.array(scan_data['C_LDA'])
    SIC = numpy.array(scan_data['self_interaction_correction'])

    # number of electronic states
    nstate = W_exact[0].shape[0]

    # Figure, axes, labels
    fig, axes = plt.subplots(1,2, figsize=(10,5))

    # Diagonal elements of indirect part of the electron repulsion operator
    axes[0].set_ylabel(r"exchange correlation / $E_h$")
    axes[0].set_xlabel(r"bond length / $\AA$")

    for i in range(0, nstate):
        line, = axes[0].plot(
            bond_length, W_exact[:,i,i] - J_Hartree[:,i,i],
            lw=2, alpha=0.5,
            label=rf"XC$_{{{i},{i}}}$")
        axes[0].plot(
            bond_length, -K_LDA[:,i,i] + C_LDA[:,i,i] + SIC,
            ls="--", color=line.get_color())

    axes[0].legend(title="$\mathbf{(c)}$ diagonal")

    # Off-diagonal elements of indirect part of electron repulsion operator
    axes[1].set_ylabel(r"exchange correlation / $E_h$")
    axes[1].set_xlabel(r"bond length / $\AA$")

    for i in range(0, nstate):
        for j in range(i+1, nstate):
            line, = axes[1].plot(
                bond_length, W_exact[:,i,j] - J_Hartree[:,i,j],
                lw=2, alpha=0.5,
                label=rf"XC$_{{{i},{j}}}$")
            axes[1].plot(
                bond_length, -K_LDA[:,i,j] + C_LDA[:,i,j],
                ls="--", color=line.get_color())

    axes[1].yaxis.set_label_position("right")
    axes[1].yaxis.set_ticks_position("right")
    axes[1].legend(title="$\mathbf{(d)}$ off-diagonal")

    # Create the invisible solid and dashed
    # black lines that are shown in the figure legend.
    solid_line = matplotlib.lines.Line2D([], [], ls="-", color="black")
    dashed_line = matplotlib.lines.Line2D([], [], ls="--", color="black")
    fig.legend(
        [solid_line, dashed_line],
        [
            r"$\text{XC}_{IJ} = \langle \Psi_I \vert \sum_{a < b} 1/r_{ab} \vert \Psi_J \rangle - \text{J}[\mathbf{D}]_{IJ}$ (exact)",
            r"$-\text{K}^{LDA}[\mathbf{D}]_{IJ} + \text{C}^{LDA}[\mathbf{D}]_{IJ} + \text{SIC}~\delta_{IJ}$"
        ],
        fontsize='large',
        frameon=False,
        loc='outside upper center',
        ncol=2,
        columnspacing=4.0
    )

    # Otherwise the x-labels are partly cut off.
    plt.subplots_adjust(bottom=0.15, wspace=0.05, left=0.1, right=0.86)

    plt.savefig("indirect_electron_repulsion_energies_lithium-fluoride.svg")
    plt.savefig("indirect_electron_repulsion_energies_lithium-fluoride.png", dpi=300)

    plt.show()
