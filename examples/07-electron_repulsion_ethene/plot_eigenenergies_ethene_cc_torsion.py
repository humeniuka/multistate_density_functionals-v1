#!/usr/bin/env python
# coding: utf-8
"""
plot the eigenenergies of the lowest few electronic states
along the C=C torsion scan of ethene.
"""
import json
import matplotlib.lines
import matplotlib.pyplot as plt
import numpy
from pyscf.data.nist import HARTREE2EV

if __name__ == "__main__":
    plt.style.use('./latex.mplstyle')
    # Load scan data
    with open('eigenenergies_ethene_cc_torsion.json', 'r') as filehandle:
        scan_data = json.load(filehandle)

    # Torsion angles in °.
    torsion_angle = numpy.array(scan_data['torsion_angle'])
    # All energies are in Hartree.
    eigenenergies = numpy.array(scan_data['eigenenergies'])

    # number of electronic states
    nstate = eigenenergies[0].shape[0]

    plt.ylabel(r"adiab. energies / eV")
    plt.xlabel(r"torsion angle / °")

    for i in range(0, nstate):
        line, = plt.plot(
            torsion_angle,
            (eigenenergies[:,i] - eigenenergies[:,0].min()) * HARTREE2EV,
            lw=2, alpha=0.5,
            label=rf"$S_{i}$")

    plt.ylim((-0.1, 16.0))
    plt.legend(reverse=True)

    #plt.savefig("eigenenergies_ethene_cc_torsion.svg")
    #plt.savefig("eigenenergies_ethene_cc_torsion.png", dpi=300)

    plt.show()
