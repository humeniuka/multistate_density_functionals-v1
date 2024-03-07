#!/usr/bin/env python
# coding: utf-8
"""
Write a table in LaTex format for the different components
contributing to the electron repulsion energies and the kinetic energies.
"""
import json
import numpy

def matrix_to_latex(matrix, title=""):
    rows, columns = matrix.shape
    assert rows == columns == 2
    tex = "    {%s} & %f & %f & %f \\\\\n" % (
        title, matrix[0,0], matrix[1,1], matrix[0,1])
    return tex

if __name__ == "__main__":
    # Load scan data
    with open('electron_repulsion_energies_lithium-fluoride.json', 'r') as filehandle:
        scan_data = json.load(filehandle)

    bond_length = numpy.array(scan_data['bond_length'])
    eigenenergies = numpy.array(scan_data['eigenenergies'])
    # electron-repulsion energies
    W_exact = numpy.array(scan_data['W_exact'])
    W_approximate = numpy.array(scan_data['W_approximate'])
    J_Hartree = numpy.array(scan_data['J_Hartree'])
    K_LDA = numpy.array(scan_data['K_LDA'])
    C_LDA = numpy.array(scan_data['C_LDA'])
    SIC = numpy.array(scan_data['self_interaction_correction'])
    # kinetic energies
    T_exact = numpy.array(scan_data['T_exact'])
    T_approximate = numpy.array(scan_data['T_approximate'])
    T_ThomasFermi = numpy.array(scan_data['T_ThomasFermi'])
    T_vonWeizsaecker = numpy.array(scan_data['T_vonWeizsaecker'])

    # The avoided crossing at 6.8 Ang corresponds to the 14th scan geometry
    index = 14-1

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

\begin{table}[ht]
  \centering
  \caption{
\normalfont
Electron repulsion and kinetic energy matrices in the basis of the exact eigenstates
$1^1\Sigma^+$ (I=0) and $2^1\Sigma^+$ (I=1) of LiF at the avoided crossing ($6.8~\AA$).
(a) Exact electron repulsion matrix $W_{IJ}^{\text{exact}}$, multistate DFT approximation
{$W[\m{D}]_{IJ} = J[\m{D}]_{IJ} -K^{LDA}[\m{D}]_{IJ} + C^{LDA}[\m{D}]_{IJ} + \text{SIC} \delta_{IJ}$},
Hartree ($J[\m{D}]_{IJ}$), exchange ($-K^{LDA}[\m{D}]_{IJ}$) and correlation ($C^{LDA}[\m{D}]_{IJ}$) matrices
and self-interaction correction for core electrons ($\text{SIC} \delta_{IJ}$).
(b) Exact kinetic energy matrix $T_{IJ}^{\text{exact}}$ and multistate DFT approximation
{$T[\m{D}]_{IJ} = T_{\text{TF}}[\m{D}]_{IJ} + \frac{1}{9} T_{\text{vW}}[\m{D}]_{IJ}$},
Thomas-Fermi ($T_{\text{TF}}[\m{D}]_{IJ}$) and von-Weizs\"{a}cker ($T_{\text{vW}}[\m{D}]_{IJ}$)
kinetic energy matrices.
All energies are in Hartree.}
\label{tbl:lithium_fluoride_electron_repulsion_and_kinetic}
"""
    # Tables for electron-repulsion
    tex += """
  \medskip

  \\begin{tabular}{SSSS}
    \\toprule
    \multicolumn{4}{c}{{\\textbf{(a)} electron repulsion, matrix elements ($I$,$J$)}} \\\\
        & {(0,0)} & {(1,1)} & {(0,1)} \\\\
    \midrule
"""
    tex += matrix_to_latex(W_exact[index], "$W_{IJ}^{\\text{exact}}$")
    tex += matrix_to_latex(W_approximate[index], "$W[\m{D}]_{IJ}$")
    tex += """
    \midrule
"""
    tex += matrix_to_latex(J_Hartree[index], "$J[\m{D}]_{IJ}$")
    tex += matrix_to_latex(-K_LDA[index], "$-K^{LDA}[\m{D}]_{IJ}$")
    tex += matrix_to_latex(C_LDA[index], "$C^{LDA}[\m{D}]_{IJ}$")
    tex += matrix_to_latex(SIC[index] * numpy.eye(W_exact[index].shape[0]), "$\\text{SIC} \delta_{IJ}$")
    tex += """
    \\bottomrule
  \end{tabular}
"""
    # Tables for kinetic energies
    tex += """
  \\bigskip

  \\begin{tabular}{SSSS}
    \\toprule
    \multicolumn{4}{c}{{\\textbf{(b)} kinetic energy, matrix elements ($I$,$J$)}} \\\\
        & {(0,0)} & {(1,1)} & {(0,1)} \\\\
    \midrule
"""
    tex += matrix_to_latex(T_exact[index], "$T_{IJ}^{\\text{exact}}$")
    tex += matrix_to_latex(T_approximate[index], "$T[\m{D}]_{IJ}$")
    tex += """
    \midrule
"""
    tex += matrix_to_latex(T_ThomasFermi[index], "$T_{\\text{TF}}[\m{D}]_{IJ}$")
    tex += matrix_to_latex(T_vonWeizsaecker[index], "$T_{\\text{vW}}[\m{D}]_{IJ}$")
    # Footer
    tex += """
    \\bottomrule
  \end{tabular}
\end{table}
%%%%%%%%%%%%%%%%%%%%%% END OF AUTO-GENERATE LATEX CODE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

    print(tex)
