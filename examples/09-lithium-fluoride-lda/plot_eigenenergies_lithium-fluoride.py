#!/usr/bin/env python
# coding: utf-8
"""
plot the eigenenergies of the lowest few electronic states
as a function of the Li-F bond length.
"""
import json
import matplotlib
import matplotlib.lines
import matplotlib.pyplot as plt
import numpy
from pyscf.data.nist import HARTREE2EV

if __name__ == "__main__":
    plt.style.use('./latex.mplstyle')
    # Load scan data
    with open('electron_repulsion_energies_lithium-fluoride.json', 'r') as filehandle:
        scan_data = json.load(filehandle)

    # Bond lengths in Å.
    bond_length = numpy.array(scan_data['bond_length'])
    # All energies are in Hartree.
    eigenenergies = numpy.array(scan_data['eigenenergies'])

    # number of electronic states
    nstate = eigenenergies[0].shape[0]

    fig = plt.figure(figsize=(4.8,4.8))

    plt.ylabel(r"adiab. energies / eV")
    plt.xlabel(r"bond length / $\AA$")

    # At the last scan geometry the bond length should be large enough
    # that the lowest state corresponds to the dissociated, neutral Li
    # and F atoms.
    dissociation_limit = eigenenergies[-1,0]

    # Zoom into the avoided crossing region.
    axes_inset = plt.gca().inset_axes([0.55, 0.55, 0.4, 0.4])
    axes_inset.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=0.5))
    axes_inset.set_xlim((6.0, 7.4))
    axes_inset.set_ylim((-0.3, 0.3))

    # Both states belong to the Σ+ irrep.
    symmetry_label = [r'$1^1\!\Sigma^+$', r'$2^1\!\Sigma^+$']
    for i in range(0, nstate):
        line, = plt.plot(
            bond_length,
            (eigenenergies[:,i] - dissociation_limit) * HARTREE2EV,
            lw=2, alpha=0.5,
            label=symmetry_label[i]
        )
        # Zoom in on avoided crossing.
        axes_inset.plot(
            bond_length,
            (eigenenergies[:,i] - dissociation_limit) * HARTREE2EV,
            lw=2, alpha=0.5,
            label=symmetry_label[i],
            color=line.get_color()
        )

    plt.legend(reverse=True, loc='lower right')

    # Otherwise the x-labels are partly cut off.
    plt.tight_layout()

    #plt.savefig("eigenenergies_lithium-fluoride.svg")
    #plt.savefig("eigenenergies_lithium-fluoride.png", dpi=300)

    plt.show()
