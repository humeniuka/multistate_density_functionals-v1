#!/usr/bin/env python
# coding: utf-8
"""
plot the exact matrix elements of the electron repulsion operator

  Iᵢⱼ = ∫dx1 ∫dx2...∫dxn Ψ*ᵢ(x1,x2,...,xn) ∑ᵦ<ᵧ 1/|rᵦ-rᵧ| Ψⱼ(x1,x2,...,xn)

and their multi-state local-density approximation

  Iᵢⱼ ≈ Jᵢⱼ[D] - Kᵢⱼ[D] + Cᵢⱼ[D] + SIC δᵢⱼ

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
    with open('electron_repulsion_energies_ethene_pyramidalization.json', 'r') as filehandle:
        scan_data = json.load(filehandle)

    # Angles are in °
    angles = numpy.array(scan_data['pyramidalization_angle'])
    eigenenergies = numpy.array(scan_data['eigenenergies'])
    I_exact = numpy.array(scan_data['I_exact'])
    I_approximate = numpy.array(scan_data['I_approximate'])

    # number of electronic states
    nstate = I_exact[0].shape[0]

    # Figure, axes, labels
    fig, axes = plt.subplots(1,2, figsize=(10,5))

    # Diagonal elements of electron repulsion operator
    # (~ classical Coulomb energies of electronic states)
    axes[0].set_ylabel(r"electron repulsion $I_{IJ}$ / $E_h$")
    axes[0].set_xlabel(r"pyramidalization angle / $^{\circ}$")

    for i in range(0, nstate):
        line, = axes[0].plot(
            angles, I_exact[:,i,i],
            lw=2, alpha=0.5,
            label=rf"$I_{{{i},{i}}}$")
        axes[0].plot(
            angles, I_approximate[:,i,i],
            ls="--", color=line.get_color())

    axes[0].legend(title="$\mathbf{(c)}$ diagonal")

    # Off-diagonal elements of electron repulsion operator
    axes[1].set_ylabel(r"electron repulsion $I_{IJ}$ / $E_h$")
    axes[1].set_xlabel(r"pyramidalization angle / $^{\circ}$")

    for i in range(0, nstate):
        for j in range(i+1, nstate):
            line, = axes[1].plot(
                angles, I_exact[:,i,j],
                lw=2, alpha=0.5,
                label=rf"$I_{{{i},{j}}}$")
            axes[1].plot(
                angles, I_approximate[:,i,j],
                ls="--", color=line.get_color())

    axes[1].yaxis.set_label_position("right")
    axes[1].yaxis.set_ticks_position("right")
    axes[1].legend(title="$\mathbf{(d)}$ off-diagonal", loc='lower left')

    # Create the invisible solid and dashed
    # black lines that are shown in the figure legend.
    solid_line = matplotlib.lines.Line2D([], [], ls="-", color="black")
    dashed_line = matplotlib.lines.Line2D([], [], ls="--", color="black")
    fig.legend(
        [solid_line, dashed_line],
        [
            r"$I_{IJ} = \langle \Psi_I \vert \sum_{m < n} 1/r_{mn} \vert \Psi_J \rangle$ (exact)",
            r"$I_{IJ} = \text{J}[\mathbf{D}]_{IJ} - \text{K}^{LSDA}[\mathbf{D}]_{IJ} + \text{C}^{LDA}[\mathbf{D}]_{IJ} + \text{SIC}~\delta_{IJ}$"
        ],
        fontsize='large',
        frameon=False,
        loc='outside upper center',
        ncol=2
    )

    # The twisted-orthogonal geometry is on the left (180°),
    # the pyramidalized geometry on the right (90°)
    axes[0].invert_xaxis()
    axes[1].invert_xaxis()

    # Otherwise the x-labels are partly cut off.
    plt.subplots_adjust(bottom=0.15, wspace=0.05, right=0.85)

    plt.savefig("electron_repulsion_energies_ethene_pyramidalization.svg")
    plt.savefig("electron_repulsion_energies_ethene_pyramidalization.png", dpi=300)

    plt.show()
