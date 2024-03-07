#!/usr/bin/env python
# coding: utf-8
"""
Write a table in LaTex format for the different components
contributing to the electron repulsion energies.
"""
import json
import numpy

def matrix_to_latex(matrix, title=""):
    """
    Generate the LaTex code for typesetting a matrix as table.
    """
    rows, columns = matrix.shape
    tex = "\\toprule\n"
    tex += "{%s} " % (title)
    for c in range(0, columns):
        tex += r"& {I=%d} " % (c)
    tex += "\\\\\n"
    tex += "\midrule\n"
    for r in range(0, rows):
        tex += r"{$J=%d$} " % (r)
        for c in range(0, columns):
            value = matrix[r,c]
            if abs(value) < 1.0e-10:
                value = 0
            tex += r"& %f" % (value)
        tex += "\\\\\n"

    return tex

if __name__ == "__main__":
    # Load scan data
    with open('electron_repulsion_energies_water_bending.json', 'r') as filehandle:
        scan_data = json.load(filehandle)

    # convert angles to degree
    angles = numpy.array(scan_data['angle']) * 180.0 / numpy.pi
    eigenenergies = numpy.array(scan_data['eigenenergies'])
    I_exact = numpy.array(scan_data['I_exact'])
    I_approximate = numpy.array(scan_data['I_approximate'])
    J_Hartree = numpy.array(scan_data['J_Hartree'])
    K_LSDA = numpy.array(scan_data['K_LSDA'])
    C_LDA = numpy.array(scan_data['C_LDA'])
    SIC = numpy.array(scan_data['self_interaction_correction'])

    # angle(HOH) = 104.2° is the 4th geometry in the scan.
    index = 3

    # Header
    tex = r"""
%%%%%%%%%%%%%%%%%%%%%% AUTO-GENERATE LATEX CODE (format_latex_table.py) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\sisetup{
round-mode = places,
round-precision = 4,
scientific-notation = fixed,
fixed-exponent = 0,
zero-decimal-to-integer
}

\begin{table}
\begin{tabular}{
  SSSSS
  }
    """
    # Tables
    tex += matrix_to_latex(I_exact[index], "$I_{IJ}^{\\text{exact}}$")
    tex += matrix_to_latex(I_approximate[index], "$I[\m{D}]_{IJ}$")
    tex += matrix_to_latex(J_Hartree[index], "$J[\m{D}]_{IJ}$")
    tex += matrix_to_latex(-K_LSDA[index], "$-K^{LSDA}[\m{D}]_{IJ}$")
    tex += matrix_to_latex(C_LDA[index], "$C^{LDA}[\m{D}]_{IJ}$")
    tex += matrix_to_latex(SIC[index] * numpy.eye(I_exact[index].shape[0]), "$\\text{SIC} \delta_{IJ}$")
    # Footer
    tex += """\\bottomrule
\end{tabular}

\caption{
\label{tbl:water_electron_repulsion}    
Electron repulsion in the lowest 4 singlet states of water ($r(OH_1)=r(OH_2)=0.958 \AA$, $\\angle(HOH) = 104.2^{\circ}$).
Exact and multistate density functional approximation for matrix of electron repulsion operator
in the basis of FCI eigenstates. The parts $\m{J}$ (Hartree energy), $-\m{K}$ (exchange), $\m{C}$ (correlation)
and SIC (self-interaction correction for core orbitals) are shown below.
All energies are in Hartree.}
\end{table}
%%%%%%%%%%%%%%%%%%%%%% END OF AUTO-GENERATE LATEX CODE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """

    print(tex)
