#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 13:07:24 2025

@author: august
"""
import numpy as np
import copy

class symetries:
    def __init__(self):
        '''
        This class contains the symmetry elements that generate the point groups
        in the main.py. For symmetry elements that are in hexangonal base, I defined 
        he_to_ca(mat_he) to convert the symmetry matrix into Cartesian
        base.
        The meaning of the variable name:
            exemple: 
                dans C2_001_he,
                C: rotation
                2: two-fold
                001: direction of the rotation axis 
                he: hexangonal base
        '''
        
        self.C3_001_he = np.array([[0,-1,0],
                                   [1,-1,0],
                                   [0,0,1]])
        
        self.C3_111_ca = np.array([[0,0,1],
                                   [1,0,0],
                                   [0,1,0]])
        
        self.C2_001_he = np.array([[-1,0,0],
                                   [0,-1,0],
                                   [0,0,1]])

        self.C2_1m10_he = np.array([[0,-1,0],
                                    [-1,0,0],
                                    [0,0,-1]])
        
        self.C2_110_ca = np.array([[0,1,0],
                                   [1,0,0],
                                   [0,0,-1]])
        
        self.C2_0m11_he = np.array([[0,-1,0],
                                    [-1,0,0],
                                    [0,0,-1]])
        
        self.m_001_he = np.array([[1,0,0],
                                  [0,1,0],
                                  [0,0,-1]])
        
        self.m_110_he = np.array([[0,1,0],
                                  [1,0,0],
                                  [0,0,1]])
        
        self.m_1m10_ca = np.array([[0,1,0],
                                   [1,0,0],
                                   [0,0,1]])
        
        self.C2_001_ca = np.array([[-1,0,0],
                                   [0,-1,0],
                                   [0,0,1]])

        self.C2_010_ca = np.array([[-1,0,0],
                                   [0,1,0],
                                   [0,0,-1]])
    
        self.S4_001_ca = np.array([[0,1,0],
                                   [-1,0,0],
                                   [0,0,-1]])

        self.C4_001_ca = np.array([[0,-1,0],
                                   [1,0,0],
                                   [0,0,1]])
        
        self.m_010_ca = np.array([[1,0,0],
                                  [0,-1,0],
                                  [0,0,1]])
        
    @staticmethod
    def he_to_ca(mat_he):
        '''
        This method serves converting symmetry matrix from hexangonal base to
        Cartesian base.
        '''
        # matrice de passage
        P_hex2ca = np.array([[1/np.cos(np.pi/6),0,0],
                             [np.tan(np.pi/6),1,0],
                             [0,0,1]])
        P_hex2ca_inv = np.linalg.inv(P_hex2ca)
        mat_ca = np.dot(P_hex2ca_inv, np.dot(mat_he, P_hex2ca))
        return mat_ca
        
        
class operations:
    def __init__(self):
        '''
        This class contains methods for implementing the Neuman's invariance principle.
        '''
        self.mat_solve = None
        self.mat_echelon = None
    
    @staticmethod
    def convertir_en_base_n(n, base_n, fill_zero=3):
        if n == 0:
            return "000"  # Si n = 0, retournez directement "000"
    
        chiffres = []
        while n > 0:
            chiffres.append(str(n % base_n))  # Ajouter le reste de la division par 3
            n //= base_n  # Division entière par base
    
        # Inverser pour obtenir l'ordre correct et ajouter des zéros à gauche
        return "".join(chiffres[::-1]).zfill(fill_zero)
        
    def reverse_ijk_map(self, ijk):
        ijk_str = self.convertir_en_base_n(ijk, 3)
        return int(ijk_str[0]), int(ijk_str[1]), int(ijk_str[2])
    
    def reverse_il_map(self, il):
        il_str = self.convertir_en_base_n(il, 6, 2)
        return int(il_str[0]), int(il_str[1])
        
    @staticmethod
    def echelonner_matrice_reduite(mat_original):
        '''
        This method solves the linear equation system of 27 variables
        via Gaussian elimination algorithm.

        Parameters
        ----------
        mat_original : numpy.array
            nx27 matrix.

        Returns
        -------
        mat : numpy.array
            echelonner form of the nx27 matrix.

        '''
        mat = copy.deepcopy(mat_original)
        lignes = len(mat)
        colonnes = len(mat[0])
    
        for i in range(min(lignes, colonnes)):
            # Trouver le pivot et échanger les lignes si nécessaire
            mat[abs(mat) < 1e-14] = 0
            pivot = i
            while pivot < lignes and mat[pivot][i] == 0:
                pivot += 1
            if pivot == lignes:
                continue
            mat[i], mat[pivot] = mat[pivot], mat[i]
    
            # Normaliser le pivot
            facteur = mat[i][i]
            mat[i] = [x / facteur for x in mat[i]]
    
            # Éliminer les éléments en dessous du pivot
            for j in range(i + 1, lignes):
                facteur = mat[j][i]
                mat[j] = [x - facteur * y for x, y in zip(mat[j], mat[i])]
    
            # Éliminer les éléments au-dessus du pivot (nouveau pour la forme réduite)
            for k in range(i - 1, -1, -1):  # Parcourir les lignes au-dessus
                facteur = mat[k][i]
                mat[k] = [x - facteur * y for x, y in zip(mat[k], mat[i])]
    
        return mat
        
    def __get_mat_solve__(self, sym_opts, tenseur_original, Kleinmann=False):
        '''
        Applying the symmetry operations on second order scuceptibility tensor.

        Parameters
        ----------
        sym_opts : list
            a list of symmetry matrices.
        tenseur_original : numpy.array
            The original second order susceptibility tensor.
        Kleinmann : boolean, optional
            Whether to include the Kleinmann's full permutation symmetry. *
            The default is False.

        Returns
        -------
        numpy.array
            echelonner form of the nx27 linear equation system.

        '''
        mat_solves = []
        for num in range(len(sym_opts)):
            mat_solves.append(np.zeros((27, 27)))
            for lmn in range(27):
                l,m,n = self.reverse_ijk_map(lmn)
                for ijk in range(27):
                    i,j,k = self.reverse_ijk_map(ijk)
                    mat_solves[num][lmn][ijk] = sym_opts[num][l][i]*sym_opts[num][m][j]*sym_opts[num][n][k]
            for i in range(27):
                mat_solves[num][i][i] -= 1
        num_syms = len(sym_opts) # numer of symmetry operations
        # add mat_solve for intrinsic permutation
        mat_solves.append(self.get_mat_solve_intrinsic_per(tenseur_original))
        num_syms += 1
        if Kleinmann:
            mat_solves.append(self.get_mat_solve_Kleinmann(tenseur_original))
            num_syms += 1
        self.mat_solve = np.array(mat_solves).reshape(num_syms*27,27)
        self.mat_echelon = self.echelonner_matrice_reduite(self.mat_solve)
        self.mat_echelon = np.round(self.mat_echelon).astype(int)
        return self.mat_echelon
        
    @staticmethod
    def permu_check(index1, index2, full_permu=True):
        '''
        check the permutation of two 3-letter strings.

        Parameters
        ----------
        index1 : str
        index2 : str
        full_permu : boolean, optional
            If False, only check the permutation of the last two letters in the string

        Returns
        -------
        boolean
            True if the permutation of the two strings are the same.

        '''
        if full_permu:
            str1 = str(index1)
            str2 = str(index2)
        else:
            str1 = str(index1)[1:]
            str2 = str(index2)[1:]           
        return sorted(str1.lower()) == sorted(str2.lower())
    
    def get_mat_solve_intrinsic_per(self, tenseur_original):
        '''
        Generate the 27x27 linear equation system to represent the intrinsic 
        permutation symmetry of the second susceptibility tensor
        '''
        mat_solve = np.zeros((27,27))
        for ijk in range(27):
            i, j, k = self.reverse_ijk_map(ijk)
            search = tenseur_original[i][j][k]
            for ijk_ in range(ijk+1, 27):
                i_, j_, k_ = self.reverse_ijk_map(ijk_)
                search_ = tenseur_original[i][j_][k_]
                if i==i_ and self.permu_check(search, search_, full_permu=False):
                    mat_solve[ijk][ijk] = 1
                    mat_solve[ijk][ijk_] = -1
                    break
        return mat_solve

    def get_mat_solve_Kleinmann(self, tenseur_original):
        '''
        Generate the 27x27 linear equation system to represent the Kleinmann's 
        full permutation symmetry of the second susceptibility tensor
        '''
        mat_solve = np.zeros((27,27))
        for ijk in range(27):
            i, j, k = self.reverse_ijk_map(ijk)
            search = tenseur_original[i][j][k]
            for ijk_ in range(ijk+1, 27):
                i_, j_, k_ = self.reverse_ijk_map(ijk_)
                search_ = tenseur_original[i_][j_][k_]
                if self.permu_check(search, search_):
                    mat_solve[ijk][ijk] = 1
                    mat_solve[ijk][ijk_] = -1
                    break
        return mat_solve


    def Neuman_Invariance(self, tenseur_original, sym_opts, Kleinmann=False):
        '''
        apply the Newmann's invariance principle to the second order susceptibility
        tensor to determine the zero elements, the equal elements, and the elements
        having the opposite sign.

        Parameters
        ----------
        tenseur_original : numpy.array
            3x3x3 dimension.
        sym_opts : list
            A list containing symmetry matrices.
        Kleinmann : Boolean, optional
            If true, then Kleinmann's full permutation symmetry is included

        Returns
        -------
        tenseur_new : numpy.array
            
        '''
        tenseur_new = np.full((3, 3, 3), np.nan)
        mat_echelon = self.__get_mat_solve__(sym_opts, tenseur_original, Kleinmann)
        # count_already_filled = 0
        # count_no_change = 0
        # count_zero = 0
        # count_same_sign = 0
        # count_oppo_sign = 0
        # count_run_through = 0
            
        for ijk in range(27):
            i,j,k = self.reverse_ijk_map(ijk)
            if not np.isnan(tenseur_new[i][j][k]):
                # count_already_filled += 1
                continue
            if sum(abs(mat_echelon[ijk])) == 0:      
                tenseur_new[i][j][k] = tenseur_original[i][j][k]
                # count_no_change += 1
                continue
            if mat_echelon[ijk][ijk] == 1 and sum(abs(mat_echelon[ijk])) == 1:
                tenseur_new[i][j][k] = 0
                # count_zero += 1
                continue
            if mat_echelon[ijk][ijk] == 1 and sum(mat_echelon[ijk]) == 0:
                ijk_ = np.where(mat_echelon[ijk] == -1)[0][0]
                i_, j_, k_ = self.reverse_ijk_map(ijk_)
                if np.isnan(tenseur_new[i_][j_][k_]):
                    tenseur_new[i][j][k] = tenseur_original[i][j][k]
                    tenseur_new[i_][j_][k_] = tenseur_original[i][j][k]
                else:
                    tenseur_new[i][j][k] = tenseur_new[i_][j_][k_]
                # count_same_sign += 1
                continue
            if mat_echelon[ijk][ijk] == 1 and sum(mat_echelon[ijk]) == 2:
                ijk_ = np.where(mat_echelon[ijk] == 1)[0][1]
                i_, j_, k_ = self.reverse_ijk_map(ijk_)
                # print('sum==2',ijk)
                if np.isnan(tenseur_new[i_][j_][k_]):
                    # print('isnan', ijk)
                    tenseur_new[i][j][k] = tenseur_original[i][j][k]
                    tenseur_new[i_][j_][k_] = -tenseur_original[i][j][k]
                else:
                    # print('not isnan', ijk)
                    tenseur_new[i][j][k] = -tenseur_new[i_][j_][k_]
                # count_oppo_sign += 1
                continue
            # count_run_through += 1
            # print('run through ijk: ', ijk, )
            # print('run through mat_echelon element: ', mat_echelon[ijk][ijk])
            # print('run through tenseur_original element: ',tenseur_original[i][j][k])
            # print('run through tenseur_new element: ',tenseur_new[i][j][k])
        # print('count_already_filled: ', count_already_filled)
        # print('count_no_change: ', count_no_change)
        # print('count_zero: ', count_zero)
        # print('count_same_sign: ', count_same_sign)
        # print('count_oppo_sign: ', count_oppo_sign)
        # print('count_run_through: ', count_run_through)
        return tenseur_new
    
    # def Neuman_Invariance(self, tenseur_original, sym_opts, Kleinmann=False):
    #     if Kleinmann == True:
    #         mat_echelon = self.__get_mat_solve__(sym_opts, tenseur_original)
    #     else:
    #         mat_echelon = self.__get_mat_solve__(sym_opts)
    #     tenseur_new = copy.deepcopy(tenseur_original)
    #     for ijk in range(27):
    #         i,j,k = self.reverse_ijk_map(ijk)
    #         # exclure l'élément nul
    #         if mat_echelon[ijk][ijk]==1 and sum(abs(mat_echelon[ijk]))==1:
    #             tenseur_new[i][j][k] = 0
    #         # exclure l'élément inchangé
    #         elif sum(abs(mat_echelon[ijk]))==0:
    #             continue
    #         # premier élément dans le trajet d'égalité
    #         elif mat_echelon[ijk][ijk] == 1:
    #             if sum(mat_echelon[ijk]) == 0:
    #                 # trouver l'élément de la meme valeur
    #                 ijk_ = np.where(mat_echelon[ijk] == -1)[0][0]
    #             else:
    #                 # trouver l'élément d'opposite valeur
    #                 ijk_ = np.where(mat_echelon[ijk] == 1)[0][1]
    #             i_, j_, k_ = self.reverse_ijk_map(ijk_)
    #             tenseur_new[i_][j_][k_] = -tenseur_new[i][j][k] * mat_echelon[ijk][ijk_]
    #         # après le premier élément dans le trajet d'égalité
    #         elif mat_echelon[ijk][ijk] == 0:
    #             ijk_ = np.where(mat_echelon[ijk] != 0)[0][0]
    #             i_, j_, k_ = self.reverse_ijk_map(ijk_)
    #             tenseur_new[i_][j_][k_] = -tenseur_new[i][j][k] * mat_echelon[ijk][ijk_]
            
    #     return tenseur_new