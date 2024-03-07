#!/usr/bin/env python
# coding: utf-8
"""
plot the eigenenergies of the lowest few electronic states
along the CH2 pyramidalization scan of ethene.
"""
import json
import matplotlib.lines
import matplotlib.pyplot as plt
import numpy
from pyscf.data.nist import HARTREE2EV

if __name__ == "__main__":
    plt.style.use('./latex.mplstyle')
    # The origin for the energy axis is the ground state energy of the planar
    # ethene molecule.
    with open('eigenenergies_ethene_cc_torsion.json', 'r') as filehandle:
        scan_data = json.load(filehandle)
    # S0 energy at a torsion angle of 0.0°.
    origin = scan_data['eigenenergies'][0][0]
    assert scan_data['torsion_angle'][0] == 0.0

    # Load scan data
    with open('eigenenergies_ethene_pyramidalization.json', 'r') as filehandle:
        scan_data = json.load(filehandle)

    # Pyramidalization angles in °.
    pyramidalization_angle = numpy.array(scan_data['pyramidalization_angle'])
    # All energies are in Hartree.
    eigenenergies = numpy.array(scan_data['eigenenergies'])

    # number of electronic states
    nstate = eigenenergies[0].shape[0]

    plt.ylabel(r"adiab. energies / eV")
    plt.xlabel(r"pyramidalization angle / °")

    for i in range(0, nstate):
        line, = plt.plot(
            pyramidalization_angle,
            (eigenenergies[:,i] - origin) * HARTREE2EV,
            lw=2, alpha=0.5,
            label=rf"$S_{i}$")

    plt.ylim((-0.1, 16.0))
    plt.gca().invert_xaxis()
    plt.gca().yaxis.set_label_position("right")
    plt.gca().yaxis.set_ticks_position("right")

    plt.legend(reverse=True)

    #plt.savefig("eigenenergies_ethene_pyramidalization.svg")
    #plt.savefig("eigenenergies_ethene_pyramidalization.png", dpi=300)

    plt.show()
