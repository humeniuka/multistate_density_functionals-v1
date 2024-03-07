#!/usr/bin/env python
# coding: utf-8
"""
plot the exact matrix elements of the kinetic energy operator

  Tᵢⱼ = <Ψᵢ|-1/2 ∑ₙ∇ₙ²|Ψⱼ>

and the multi-state local spin density approximation, which is a
combination of the Thomas-Fermi (TF) with the von-Weizsäcker (vW)
kinetic functionals,

  T[D(r)]ᵢⱼ ≈ T_TF[D]ᵢⱼ + 1/9 T_vW[D]ᵢⱼ

            = 3/10 (6π²)²ᐟ³ ∫ [(Dᵅ(r)⁵ᐟ³)ᵢⱼ + (Dᵝ(r)⁵ᐟ³)ᵢⱼ] dr + 1/9 ∫ 1/8 ∑ₖ∑ₗ ∇Dᵢₖ D⁻¹ₖₗ ∇Dₗⱼ

The  the diagonal (i=j) and off-diagonal (i≠j) elements for all
scan geometries are plotted separately.
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
    T_exact = numpy.array(scan_data['T_exact'])
    T_approximate = numpy.array(scan_data['T_approximate'])

    # number of electronic states
    nstate = T_exact[0].shape[0]

    # Figure, axes, labels
    fig, axes = plt.subplots(1,2, figsize=(10,5))

    # Diagonal elements of kinetic energy operator.
    axes[0].set_ylabel(r"kinetic energy $T_{II}$ / $E_h$")
    axes[0].set_xlabel(r"bond length / $\AA$")

    for i in range(0, nstate):
        line, = axes[0].plot(
            bond_length, T_exact[:,i,i],
            lw=2, alpha=0.5,
            label=rf"$T_{{{i},{i}}}$")
        axes[0].plot(
            bond_length, T_approximate[:,i,i],
            ls="--", color=line.get_color())

    axes[0].legend(title="$\mathbf{(e)}$ diagonal")

    # Off-diagonal elements kinetic energy operator
    axes[1].set_ylabel(r"kinetic energy $T_{IJ}$ / $E_h$")
    axes[1].set_xlabel(r"bond length / $\AA$")

    for i in range(0, nstate):
        for j in range(i+1, nstate):
            line, = axes[1].plot(
                bond_length, T_exact[:,i,j],
                lw=2, alpha=0.5,
                label=rf"$T_{{{i},{j}}}$")
            axes[1].plot(
                bond_length, T_approximate[:,i,j],
                ls="--", color=line.get_color())

    axes[1].yaxis.set_label_position("right")
    axes[1].yaxis.set_ticks_position("right")
    axes[1].legend(title="$\mathbf{(f)}$ off-diagonal")

    # Create the invisible solid and dashed
    # black lines that are shown in the figure legend.
    solid_line = matplotlib.lines.Line2D([], [], ls="-", color="black")
    dashed_line = matplotlib.lines.Line2D([], [], ls="--", color="black")
    fig.legend(
        [solid_line, dashed_line],
        [
            r"$T_{IJ} = \langle \Psi_I \vert -\frac{1}{2} \sum_m \nabla_m^2 \vert \Psi_J \rangle$ (exact)",
            r"$T_{IJ} = T_{TF}[\mathbf{D}]_{IJ} + \frac{1}{9} T_{vW}[\mathbf{D}]_{IJ}$"
        ],
        fontsize='large',
        frameon=False,
        loc='outside upper center',
        ncol=2
    )

    # Otherwise the x-labels are partly cut off.
    plt.subplots_adjust(bottom=0.15, wspace=0.05, left=0.1, right=0.86)

    plt.savefig("kinetic_energies_lithium-fluoride.svg")
    plt.savefig("kinetic_energies_lithium-fluoride.png", dpi=300)

    plt.show()
