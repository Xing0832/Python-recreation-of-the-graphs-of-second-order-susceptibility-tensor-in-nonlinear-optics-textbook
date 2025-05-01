# Python-recreation-of-the-graphs-of-second-order-susceptibility-tensor-in-nonlinear-optics-textbook
Python implementation of Neumann's invariance principle to determine the non-zero elements, equal elements and elements of opposite signs in the second order susceptibility tensor.

In chapter 1 of the textbook of Nonlinear Optics by Robert W. Boyd, the author introduced the form of contracted second order susceptibility tensor represented in graphs (FIGURE 1.5.3 in the third edition of the book) originally made by Zernike and Midwinter in 1973. These graphs aim to show the non-zero elements, equal elements and elements of the opposite sign in the tensor due to the symmetry properties of the medium. However, for learners who want to recreate those graphs themselves, some arbitrary yet nuanced information, such as the chosen base of the symmetry matrix and the inclusion of the instrinsic permutation, are not explicitly provided in the textbook.

This repository is the python code to recreate all these graphs. 

The file class_symetry_operation.py contains two classes: symetries and operations. The symetries class contains all the symmetry matrices that generate the point groups originally represented in the textbook. operations class contains methods that perform the Neumann's invariance principle.

The file main.py is the script to select the point group and visualise the effect of the point group on the tensor elements.
