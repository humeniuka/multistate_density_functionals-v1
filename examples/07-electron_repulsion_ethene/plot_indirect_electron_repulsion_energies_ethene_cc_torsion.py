#!/usr/bin/env python
# coding: utf-8
"""
plot the exact matrix elements of the electron repulsion operator

  Iᵢⱼ = ∫dx1 ∫dx2...∫dxn Ψ*ᵢ(x1,x2,...,xn) ∑ᵦ<ᵧ 1/|rᵦ-rᵧ| Ψⱼ(x1,x2,...,xn)

i.e.

  Iᵢⱼ - Jᵢⱼ[D]

and compare it with the multistate local density approximation (LDA) for
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
    with open('electron_repulsion_energies_ethene_cc_torsion.json', 'r') as filehandle:
        scan_data = json.load(filehandle)

    # Angles are in °.
    angles = numpy.array(scan_data['torsion_angle'])
    eigenenergies = numpy.array(scan_data['eigenenergies'])
    I_exact = numpy.array(scan_data['I_exact'])
    I_approximate = numpy.array(scan_data['I_approximate'])
    J_Hartree = numpy.array(scan_data['J_Hartree'])
    K_LSDA = numpy.array(scan_data['K_LSDA'])
    C_LDA = numpy.array(scan_data['C_LDA'])
    SIC = numpy.array(scan_data['self_interaction_correction'])

    # number of electronic states
    nstate = I_exact[0].shape[0]

    # Figure, axes, labels
    fig, axes = plt.subplots(1,2, figsize=(10,5))

    # Diagonal elements of indirect part of electron repulsion operator
    # (~ exchange-correlation energies of electronic states)
    axes[0].set_ylabel(r"exchange correlation / $E_h$")
    axes[0].set_xlabel(r"torsion angle / $^{\circ}$")

    for i in range(0, nstate):
        line, = axes[0].plot(
            angles, I_exact[:,i,i] - J_Hartree[:,i,i],
            lw=2, alpha=0.5,
            label=rf"XC$_{{{i},{i}}}$")
        axes[0].plot(
            angles, -K_LSDA[:,i,i] + C_LDA[:,i,i] + SIC,
            ls="--", color=line.get_color())

    axes[0].legend(title="$\mathbf{(a)}$ diagonal")

    # Off-diagonal elements of indirect part of electron repulsion operator
    axes[1].set_ylabel(r"exchange correlation / $E_h$")
    axes[1].set_xlabel(r"torsion angle / $^{\circ}$")

    for i in range(0, nstate):
        for j in range(i+1, nstate):
            line, = axes[1].plot(
                angles, I_exact[:,i,j] - J_Hartree[:,i,j],
                lw=2, alpha=0.5,
                label=rf"XC$_{{{i},{j}}}$")
            axes[1].plot(
                angles, -K_LSDA[:,i,j] + C_LDA[:,i,j],
                ls="--", color=line.get_color())

    axes[1].yaxis.set_label_position("right")
    axes[1].yaxis.set_ticks_position("right")
    axes[1].legend(title="$\mathbf{(b)}$ off-diagonal", loc='center left')

    # Create the invisible solid and dashed
    # black lines that are shown in the figure legend.
    solid_line = matplotlib.lines.Line2D([], [], ls="-", color="black")
    dashed_line = matplotlib.lines.Line2D([], [], ls="--", color="black")
    fig.legend(
        [solid_line, dashed_line],
        [
            r"$\text{XC}_{IJ} = \langle \Psi_I \vert \sum_{m < n} 1/r_{mn} \vert \Psi_J \rangle - \text{J}[\mathbf{D}]_{IJ}$ (exact)",
            r"$-\text{K}^{LSDA}[\mathbf{D}]_{IJ} + \text{C}^{LDA}[\mathbf{D}]_{IJ} + \text{SIC}~\delta_{IJ}$"
        ],
        fontsize='large',
        frameon=False,
        loc='outside upper center',
        ncol=2
    )

    # Otherwise the x-labels are partly cut off.
    plt.subplots_adjust(bottom=0.15, wspace=0.05, right=0.85)

    plt.savefig("indirect_electron_repulsion_energies_ethene_cc_torsion.svg")
    #plt.savefig("indirect_electron_repulsion_energies_ethene_cc_torsion.png", dpi=300)

    plt.show()
