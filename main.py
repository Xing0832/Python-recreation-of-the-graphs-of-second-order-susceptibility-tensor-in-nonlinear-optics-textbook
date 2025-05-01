#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 17:52:47 2025

@author: august
"""
import numpy as np
import matplotlib.pyplot as plt
from class_symetry_operation import symetries, operations

obj_syms = symetries()
obj_opts = operations()

'''
##############################################################################
define the point group symmetries
##############################################################################
'''
# for example, S4 plus C2 with respect to z-axis
sym_opts_class_S4 = [obj_syms.C2_001_ca, obj_syms.S4_001_ca]
sym_opts_class_Cs = [obj_syms.m_010_ca]
sym_opts_class_C2 = [obj_syms.C2_010_ca]
sym_opts_class_C3 = [obj_syms.he_to_ca(obj_syms.C3_001_he)]
sym_opts_class_D2 = [obj_syms.C2_001_ca, obj_syms.C2_010_ca]
sym_opts_class_C2v = [obj_syms.C2_001_ca, obj_syms.m_010_ca]
sym_opts_class_C3v = [obj_syms.he_to_ca(obj_syms.C3_001_he), 
                      obj_syms.he_to_ca(obj_syms.m_110_he)]
sym_opts_class_C3h = [obj_syms.he_to_ca(obj_syms.C3_001_he), 
                      obj_syms.he_to_ca(obj_syms.m_001_he)]
sym_opts_class_D3h = [obj_syms.he_to_ca(obj_syms.C3_001_he), 
                      obj_syms.he_to_ca(obj_syms.m_001_he),
                      obj_syms.he_to_ca(obj_syms.m_110_he)]
sym_opts_class_C4 = [obj_syms.C2_001_ca, obj_syms.C4_001_ca]
sym_opts_class_C6 = [obj_syms.he_to_ca(obj_syms.C3_001_he), obj_syms.C2_001_ca]
sym_opts_class_C6v = [obj_syms.he_to_ca(obj_syms.C3_001_he), 
                      obj_syms.C2_001_ca,
                      obj_syms.he_to_ca(obj_syms.m_110_he)]
sym_opts_class_C4v = [obj_syms.C2_001_ca, obj_syms.C4_001_ca, obj_syms.m_010_ca]
sym_opts_class_D3 = [obj_syms.he_to_ca(obj_syms.C3_001_he),
                     obj_syms.he_to_ca(obj_syms.C2_1m10_he)]
sym_opts_class_D2d = [obj_syms.C2_001_ca, obj_syms.S4_001_ca, obj_syms.C2_010_ca]
sym_opts_class_Td = [obj_syms.C2_001_ca, obj_syms.C2_010_ca, obj_syms.C3_111_ca, 
                     obj_syms.m_1m10_ca]
sym_opts_class_T = [obj_syms.C2_001_ca, obj_syms.C2_010_ca, obj_syms.C3_111_ca]
sym_opts_class_O = [obj_syms.C2_001_ca, obj_syms.C2_010_ca, obj_syms.C3_111_ca,
                    obj_syms.C2_110_ca]


'''
##############################################################################
# create the second order susceptibility tensor and apply Neuman's invariance
principle
##############################################################################
'''
tensor_original = np.zeros((3,3,3))
# fill the tensor with numbers that represent the subscript of tensor elements
for ijk in range(27):
    i,j,k = obj_opts.reverse_ijk_map(ijk)
    tensor_original[i][j][k] = (k+1) + (j+1)*10 + (i+1)*100

# apply the Neuman's invariance principle
tensor_new = obj_opts.Neuman_Invariance(tensor_original, sym_opts_class_D3, Kleinmann=False)

'''
##############################################################################
contract the tensors and visualise
##############################################################################
'''
def contraction(j,k):
    if j==0 and k==0:
        l = 0
    if j==1 and k==1:
        l=1
    if j==2 and k==2:
        l=2
    if (j==1 and k==2) or (j==2 and k==1):
        l=3
    if (j==0 and k==2) or (j==2 and k==0):
        l=4
    if (j==0 and k==1) or (j==1 and k==0):
        l=5
    return l


tensor_contra_ijk = np.full((3, 6), '', dtype=object)
tensor_contra_il = np.full((3, 6), '', dtype=object)
for ijk in range(27):
    i,j,k = obj_opts.reverse_ijk_map(ijk)
    l = contraction(j,k)
    if tensor_new[i][j][k] == 0:
        tensor_contra_il[i][l] = '00'
        tensor_contra_ijk[i][l] += '000,'
    else:
        tensor_contra_ijk[i][l] += str(round(tensor_new[i][j][k])) + ','
        if tensor_contra_ijk[i][l][0] != '-':
            first_index = 0
            first_letter = ''
        else:
            first_index = 1
            first_letter = '-'
        i_ = tensor_contra_ijk[i][l][first_index+0]
        j_ = tensor_contra_ijk[i][l][first_index+1]
        k_ = tensor_contra_ijk[i][l][first_index+2]
        l_ = contraction(int(j_)-1,int(k_)-1) + 1
        tensor_contra_il[i][l] = first_letter + i_ + str(l_)       
        

# draw the figure
plt.figure()

# plotter les points
for il in range(18):
    i,l = obj_opts.reverse_il_map(il)
    if tensor_contra_il[i][l] == '00':
        markersize = 12
    else:
        markersize = 128
    plt.scatter(l, i, marker='o', color='black', 
                facecolor='black', s=markersize, zorder=1)
plt.gca().invert_yaxis() 


# count the ocurrence of the absolute value of elements
elements, counts = np.unique(abs(tensor_contra_il.astype(int)), return_counts=True)

# choose the non-zero reocurring elements
doublons = elements[(counts > 1)*(elements != 0)]

# extract the position of the non-zero reocurring elements
dict_pos =dict(zip(doublons, [np.where(abs(tensor_contra_il.astype(int))==key) for key in doublons]))

# connect the non-zero reocurring elements
for key in dict_pos:
    x_triee, y_triee = zip(*sorted(zip(dict_pos[key][1], dict_pos[key][0])))
    plt.plot(list(x_triee), list(y_triee), marker='o', markersize=12, zorder=2)

# mark the negative points
neg_points = np.where(tensor_contra_il.astype(int)<0)
plt.scatter(neg_points[1],neg_points[0], marker='o', color='white', 
            facecolor='white', s=80, zorder=3)

