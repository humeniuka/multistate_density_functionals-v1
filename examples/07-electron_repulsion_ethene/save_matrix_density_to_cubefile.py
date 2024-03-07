#!/usr/bin/env python
# coding: utf-8
"""
Save the CASSCF matrix density to a series of cube files.
"""
import numpy
import pyscf.tools.cubegen
import pyscf.tools.molden

from msdft.MultistateMatrixDensity import MultistateMatrixDensityCASSCF


if __name__ == "__main__":
    # Ethene
    # The experimental HCH angle is 117.6°.
    angleHCH = 117.6 * numpy.pi/180.0
    # The experimental CH bond length is 1.086 Å.
    rCH = 1.086
    # The experimental carbon-carbon bond length of ethene (NIST) is 1.339 Å.
    rCC = 1.339
    # The hydrogens lie in the xy plane.
    xH = rCH * numpy.cos(angleHCH/2.0)
    yH = rCH * numpy.sin(angleHCH/2.0)
    # The C=C bond is parallel to the x-axis.
    # The =CH2 group is rotated around the x-axis.
    # C=C torsion angle
    torsion_angle = 0.0 * numpy.pi/180.0
    # The pyramidalized CH2 group is rotated around the y-axis.
    pyramidalization_angle = 180.0 * numpy.pi/180.0

    mol = pyscf.gto.M(
        atom = f"""
        C {-rCC/2} 0 0
        C { rCC/2} 0 0
        H { rCC/2+xH} { numpy.cos(torsion_angle) * yH} { numpy.sin(torsion_angle) * yH}
        H { rCC/2+xH} {-numpy.cos(torsion_angle) * yH} {-numpy.sin(torsion_angle) * yH}
        H {-rCC/2+numpy.cos(pyramidalization_angle) * xH} { yH} {-numpy.sin(pyramidalization_angle) * xH}
        H {-rCC/2+numpy.cos(pyramidalization_angle) * xH} {-yH} {-numpy.sin(pyramidalization_angle) * xH}
        """,
        # small basis set
        basis = '6-31g*',
        charge = 0,
        # singlet
        spin = 0)

    # Save geometry in molden format.
    with open("/tmp/geometry.molden", "w") as molden_file:
        pyscf.tools.molden.header(mol, molden_file, ignore_h=False)

    # Compute lowest 3 singlet states.
    nstate = 3

    # Compute D(r) from CASSCF.
    msmd = MultistateMatrixDensityCASSCF.create_matrix_density(
        mol,
        nstate=nstate,
        # The complete active space consists of 2 electrons in 2 orbitals
        ncas=2,
        nelecas=2)

    # Write the (transition) densities D(r) to cube files.
    cube = pyscf.tools.cubegen.Cube(mol)
    # Evaluate D(r) on the rectangular grid of the cube.
    coords = cube.get_coords()
    D, _, _ = msmd.evaluate(coords)

    # Loop over electronic states.
    for istate in range(0, nstate):
        for jstate in range(istate, nstate):
            if istate == jstate:
                comment = f"Density of state {istate}"
            else:
                comment = f"Transition density between states {istate} and {jstate}"
            # spin up Dᵢⱼ(r)
            field = numpy.reshape(D[0,istate,jstate,:], (cube.nx, cube.ny, cube.nz))
            cube.write(
                field,
                f'/tmp/matrix_density_{istate}_{jstate}_up.cube',
                comment=comment + " (spin up)")
            # spin down Dᵢⱼ(r)
            field = numpy.reshape(D[1,istate,jstate,:], (cube.nx, cube.ny, cube.nz))
            cube.write(
                field,
                f'/tmp/matrix_density_{istate}_{jstate}_down.cube',
                comment=comment + " (spin down)")
            # Total spin density
            field = numpy.reshape(
                D[0,istate,jstate,:]+D[1,istate,jstate,:], (cube.nx, cube.ny, cube.nz))
            cube.write(
                field,
                f'/tmp/matrix_density_{istate}_{jstate}.cube',
                comment=comment + " (spin up + spin down)")

    print("Cube files were saved to /tmp/matrix_density_[istate]_[jstate]_[spin].cube")
