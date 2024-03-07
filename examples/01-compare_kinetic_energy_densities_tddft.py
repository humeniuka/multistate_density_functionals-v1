#!/usr/bin/env python
# coding: utf-8
"""
compare kinetic energy densities KED(r) computed with the von-Weizsaecker functional
and the Thomas-Fermi functional with the exact values for some test molecules.
"""

import numpy

import pyscf.gto
import pyscf.scf
import pyscf.tddft

from msdft.KineticOperatorFunctional import LSDAThomasFermiFunctional
from msdft.KineticOperatorFunctional import LSDAVonWeizsaecker1eFunctional
from msdft.KineticOperatorFunctional import LSDAVonWeizsaeckerFunctional
from msdft.MultistateMatrixDensity import MultistateMatrixDensityTDDFT


""" dictionary with atoms/molecules to run the tests on """
molecules = {
    # H2
    'hydrogen molecule': pyscf.gto.M(
        atom = 'H 0 0 -0.375; H 0 0 0.375',
        basis = '6-31g',
        # singlet
        spin = 0),
    # 3-electron systems, one unpaired spin
    'lithium atom': pyscf.gto.M(
        atom = 'Li 0 0 0',
        basis = '6-31g',
        # doublet
        spin = 1),
    # noble gas atom
    'neon atom': pyscf.gto.M(
        atom = 'Ne 0 0 0',
        basis = '6-31g',
        # singlet
        spin = 0),
    # LiH
    'lithium hydride': pyscf.gto.M(
        atom = 'Li 0 0 -0.79745; H 0 0 0.79745',
        basis = '6-31g',
        # singlet
        spin = 0),
    # Li2
    'lithium diatomic': pyscf.gto.M(
        atom = 'Li 0 0 -1.335; Li 0 0 1.335',
        basis = '6-31g',
        # singlet
        spin = 0),
    # H2O
    'water': pyscf.gto.M(
        atom = 'O 0.0 0.0 0.1173; H 0.0 0.7572 -0.4692; H 0.0 -0.7572 -0.4692',
        basis = '6-31g',
        # singlet
        spin = 0),
    # naphthalene
    'naphthalene': pyscf.gto.M(
        atom = '01-naphthalene.xyz',
        basis = '6-31g',
        # singlet
        spin = 0).build()
}


def compare_kinetic_energy_densities_1d(mol, nstate=2):
    """
    The kinetic energy density is plotted for different functionals
    and is compared with the exact one.
    
    :param nstate: Number of electronic states in the subspace.
       The TD-DFT problem is solved for the lowest nstate-1 states.
    :type nstate: int > 0
    """
    # compute D(r) from full TD-DFT
    msmd = MultistateMatrixDensityTDDFT.create_matrix_density(mol, nstate=nstate)
    
    # Plot T(0,0,z), cut along z-axis
    Ncoord = 5000
    coords = numpy.zeros((Ncoord, 3))
    r = numpy.linspace(-6.0, 6.0, Ncoord)
    coords[:,1] = 1.3605839999999998
    coords[:,2] = r
        
    # Functionals for kinetic energy matrix.
    kinetic_vW = LSDAVonWeizsaeckerFunctional(mol)
    kinetic_TF = LSDAThomasFermiFunctional(mol)

    # Evalute kinetic energy density along the cut ...
    # ... with the approximate functionals from the matrix density
    KED_vW = kinetic_vW.kinetic_energy_density(msmd, coords)
    KED_TF = kinetic_TF.kinetic_energy_density(msmd, coords)
    # ... and exactly from the wavefunction.
    KED_lap, KED_gg = msmd.kinetic_energy_density(coords)

    # Sum over spin.
    KED_vW = numpy.sum(KED_vW, axis=0)
    KED_TF = numpy.sum(KED_TF, axis=0)
    
    KED_lap = numpy.sum(KED_lap, axis=0)
    KED_gg = numpy.sum(KED_gg, axis=0)

    # Plot KED(r).
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,2)

    # The first row is for the von-Weizsaecker functional,
    # the second one for the Thomas-Fermi functional.
    for row in [0,1]:
        axes[row,0].set_xlabel(r"r / $a_0$")
        axes[row,0].set_ylabel(r"state KED / Hartree")
    
        axes[row,1].set_xlabel(r"r / $a_0$")
        axes[row,1].set_ylabel(r"transition KED / Hartree")

    for column in [0,1]:
        axes[0,column].set_title(r"von Weizsäcker")
        axes[1,column].set_title(r"Thomas-Fermi")
        
    # Plot kinetic energy density between different states.
    for istate in range(0, nstate):
        for jstate in range(istate, nstate):
            label = r"KED$_{%d,%d}$" % (istate, jstate)
            # Diagonal (state) and off-diagonal (transition) kinetic
            # energy densities are plotted separately.
            if istate == jstate:
                column = 0
            else:
                column = 1

            # exact KED as reference
            """
            axes[0,column].plot(
                r, KED_lap[istate,jstate,:],
                lw=3, alpha=0.25, label=label+" exact (-1/2 ∇²f)")
            axes[1,column].plot(
                r, KED_lap[istate,jstate,:],
                lw=3, alpha=0.25, label=label+" exact (-1/2 ∇²f)")
            """
            line, = axes[0,column].plot(
                r, KED_gg[istate,jstate,:],
                lw=3, alpha=0.25,
                label=label+" exact (1/2 ∇f ∇f)")
            axes[1,column].plot(
                r, KED_gg[istate,jstate,:],
                lw=3, color=line.get_color(), alpha=0.25,
                label=label+" exact (1/2 ∇f ∇f)")

            # approximate KED
            axes[0,column].plot(
                r, KED_vW[istate,jstate,:],
                lw=1, color=line.get_color(),
                ls='--', label=label+" von-Weizsäcker")
            axes[1,column].plot(
                r, KED_TF[istate,jstate,:],
                lw=1, color=line.get_color(),
                ls="-.", label=label+" Thomas-Fermi")

    for row in [0,1]:
        for column in [0,1]:
            axes[row,column].legend()

    plt.show()


def compare_kinetic_energy_densities_2d(mol, nstate=2, istate=0, jstate=0):
    """
    The kinetic energy density is plotted for different functionals
    in the yz-plane and is compared with the exact one.
    
    :param nstate: Number of electronic states in the subspace.
       The TD-DFT problem is solved for the lowest nstate-1 states.
    :type nstate: int > 0

    :param istate, jstate:
       The indices of the electronic states (0 - ground state) for which
       the transition density (istate!=jstate) or the state density (istate==jstate)
       should be plotted.
    :type istate, jstate: int
    """
    # compute D(r) from full TD-DFT
    msmd = MultistateMatrixDensityTDDFT.create_matrix_density(mol, nstate=nstate)
    
    # Plot T(0,y,z) in the yz plane.
    ny = 200
    nz = 400
    y = numpy.linspace(-6.0, 6.0, ny)
    z = numpy.linspace(-6.0, 6.0, nz)
    ygrid_2d, zgrid_2d = numpy.meshgrid(y,z)

    coords = numpy.zeros((ny*nz, 3))
    coords[:,1] = ygrid_2d.ravel()
    coords[:,2] = zgrid_2d.ravel()
        
    # Functionals for kinetic energy matrix.
    kinetic_vW = LSDAVonWeizsaeckerFunctional(mol)
    kinetic_TF = LSDAThomasFermiFunctional(mol)

    # Evalute kinetic energy density along the cut ...
    # ... with the approximate functionals from the matrix density
    KED_vW = kinetic_vW.kinetic_energy_density(msmd, coords)
    KED_TF = kinetic_TF.kinetic_energy_density(msmd, coords)
    # ... and exactly from the wavefunction.
    KED_lap, KED_gg = msmd.kinetic_energy_density(coords)

    # Sum over spin.
    KED_vW = numpy.sum(KED_vW, axis=0)
    KED_TF = numpy.sum(KED_TF, axis=0)
    
    KED_lap = numpy.sum(KED_lap, axis=0)
    KED_gg = numpy.sum(KED_gg, axis=0)

    if istate == jstate:
        # The kinetic energy is very large around the nuclei, plot the log
        KED_vW = numpy.log(KED_vW)
        KED_TF = numpy.log(KED_TF)
        KED_lap = numpy.log(KED_lap)
        KED_gg = numpy.log(KED_gg)

    # Plot KED(r).
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3)

    # The first row is for the von-Weizsaecker functional,
    # the second one for the Thomas-Fermi functional.
    for row in [0,1,2]:
        axes[row].set_ylabel(r"y / $a_0$")
        axes[row].set_xlabel(r"z / $a_0$")

    label = r"KED$_{%d,%d}$" % (istate, jstate)
    axes[1].set_title(r"exact (1/2 ∇f ∇f) "+label)
    axes[1].set_title(r"von Weizsäcker "+label)
    axes[2].set_title(r"Thomas-Fermi "+label)

    # exact KED as reference
    axes[0].imshow(
        numpy.reshape(KED_gg[istate,jstate,:], (nz, ny)).transpose())

    # approximate KED
    axes[1].imshow(
        numpy.reshape(KED_vW[istate,jstate,:], (nz, ny)).transpose())

    axes[2].imshow(
        numpy.reshape(KED_TF[istate,jstate,:], (nz, ny)).transpose())

    plt.show()


if __name__ == "__main__":
    #compare_kinetic_energy_densities_1d(molecules['hydrogen molecule'], nstate=2)
    #compare_kinetic_energy_densities_1d(molecules['lithium hydride'], nstate=2)
    #compare_kinetic_energy_densities_1d(molecules['lithium diatomic'], nstate=3)
    #compare_kinetic_energy_densities_1d(molecules['water'], nstate=4)
    #compare_kinetic_energy_densities_1d(molecules['naphthalene'], nstate=3)
    compare_kinetic_energy_densities_2d(molecules['naphthalene'], nstate=2, istate=0, jstate=0)
