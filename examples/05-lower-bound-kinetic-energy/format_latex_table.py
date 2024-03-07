#!/usr/bin/env python
# coding: utf-8
"""
Write a table in LaTex format for comparing the lower bounds
of the subspace kinetic energy with the exact one.
"""
import numpy
import pandas

def matrix_to_latex(dataframe):
    """
    Generate the LaTex code for typesetting a matrix as table.
    """
    columns = list(dataframe.columns)
    tex = "\\toprule\n"
    tex += r" {Molecule} & {Basis} & {SoS bound} & {Inv bound} & {Exact}"
    tex += "\\\\\n"
    tex += r" & &  {Eqn.~\ref{eqn:lower_bound_kinetic_sum_over_states}} & {Eqn.~\ref{eqn:lower_bound_kinetic_subspace_invariant}} & "
    tex += "\\\\\n"
    tex += "\midrule\n"
    for r in range(0, len(dataframe)):
        tex += r"{%s} & {%s} & %f & %f & %f " % (
            dataframe['molecule'][r],
            dataframe['basis'][r],
            dataframe['bound 1'][r],
            dataframe['bound 2'][r],
            dataframe['exact'][r]
        )
        tex += "\\\\\n"

    return tex

if __name__ == "__main__":
    # Load data
    dataframe = pandas.read_csv('kinetic_energy_bounds.csv')

    # Header
    tex = r"""
%%%%%%%%%%%%%%%%%%%%%% AUTO-GENERATE LATEX CODE (format_latex_table.py) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\sisetup{
round-mode = places,
round-precision = 3,
scientific-notation = fixed,
fixed-exponent = 0,
zero-decimal-to-integer
}

\begin{table}
\begin{tabular}{
  SSSSS
  }
    """
    # Table
    tex += matrix_to_latex(dataframe)
    # Footer
    tex += """\\bottomrule
\end{tabular}

\caption{
\label{tbl:lower_bounds_kinetic_energy}
\normalfont
The exact subspace kinetic energy $\\frac{1}{N} \sum_{I=1}^{N} T_{II}$ (\\textbf{Exact}) is bounded from below by
the \\textbf{S}um \\textbf{O}ver \\textbf{S}tates bound $\\frac{1}{N} \\frac{1}{8} \sum_{I=1}^{N} \int \\frac{\\vert \\nabla D_{II} \\vert^2}{D_{II}}$
(\\textbf{SoS bound} Eqn.~\\ref{eqn:lower_bound_kinetic_sum_over_states}) and by the subspace \\textbf{Inv}ariant bound
$\\frac{1}{8} \int \\frac{\\vert \\nabla \\rho_V \\vert^2}{\\rho_V}$
(\\textbf{Inv bound} Eqn.~\\ref{eqn:lower_bound_kinetic_subspace_invariant})
The exact wavefunctions for the $N=4$ lowest excited states ($N=2$ for hydrogen atom with small basis sets)
are calculated with full configuration interaction for a few small atoms and molecules.
All energies are in Hartree.}
\end{table}
%%%%%%%%%%%%%%%%%%%%%% END OF AUTO-GENERATE LATEX CODE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """

    print(tex)
