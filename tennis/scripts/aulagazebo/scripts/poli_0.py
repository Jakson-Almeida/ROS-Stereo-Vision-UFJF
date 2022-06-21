#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from multipledispatch import dispatch

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class Gauss_Elimination:
    def __init__(self, mat_a, mat_b):
        self.mat_a = None
        self.mat_x = None
        self.mat_b = None
        if(mat_a.shape[0] != mat_a.shape[1] or mat_a.getNumColunas() == 0):
            return
        if(mat_b.getNumLinhas() == 1):
            mat_b.transposta()
        self.mat_a = mat_a.copy()
        self.mat_x = np.arange(mat_a.getNumLinhas(), 1)
        self.mat_b = mat_b.copy()
        self.triangleMatrix()
        self.resolveSuperTriangleMatrix()
    
    # Para a eliminação de Gauss chegaremos a uma matriz triangular usando o pivotamento
    def  triangleMatrix(self):
        max = None
        index = 0
        for k in range(self.mat_a.getNumColunas() - 1):
            max = -9999999
            index = k
            for i in range(self.mat_a.getNumLinhas()):
                if (np.abs(self.mat_a.mat[i][k]) > max and self.mat_a.mat[i][k] != 0):
                    max = self.mat_a.mat[i][k]
                    index = i
        self.mat_a.swapLines(k + 1, index + 1)
        self.mat_b.swapLines(k + 1, index + 1)
        for i in range(self.mat_a.getNumLinhas()):
            m = self.mat_a.mat[i][k] / self.mat_a.mat[k][k]
            for j in range(self.mat_a.getNumColunas()):
                self.mat_a.mat[i][j] = self.mat_a.mat[i][j] - m*self.mat_a.mat[k][j]
                self.mat_b.mat[i][0] = self.mat_b.mat[i][0] - m*self.mat_b.mat[k][0]
        # mat_a.printMatrix()
        # mat_b.printMatrix()
        # println()
    
    #@dispatch(Matrix, Matrix)
    def triangleMatrix(self, mat_a, mat_b):
        max = None
        index = 0
        for k in range(mat_a.getNumColunas() - 1):
            max = -9999999
            index = k
            for i in range(mat_a.getNumLinhas()):
                if (np.abs(mat_a.mat[i][k]) > max and mat_a.mat[i][k] != 0):
                    max = mat_a.mat[i][k]
                    index = i
        mat_a.swapLines(k + 1, index + 1)
        mat_b.swapLines(k + 1, index + 1)
        for i in range(mat_a.getNumLinhas()):
            m = mat_a.mat[i][k] / mat_a.mat[k][k]
            for j in range(mat_a.getNumColunas()):
                mat_a.mat[i][j] = mat_a.mat[i][j] - m*mat_a.mat[k][j]
                mat_b.mat[i][0] = mat_b.mat[i][0] - m*mat_b.mat[k][0]
    
    def resolveSuperTriangleMatrix(self):
        n = self.mat_x.getNumLinhas() - 1
        self.mat_x.mat[n][0] = self.mat_b.mat[n][0] / self.mat_a.mat[n][n]
        for i in (n - 1, -1, -1):
            sum = 0
            for j in (i + 1, n, 1):
                sum += self.mat_a.mat[i][j]*self.mat_x.mat[j][0]
                self.mat_x.mat[i][0] = (self.mat_b.mat[i][0] - sum) / self.mat_a.mat[i][i]
        # mat_x.printMatrix()
    
    #@dispatch(Matrix, Matrix)
    def resolveSuperTriangleMatrix(self, mat_a, mat_b):
        mat_x = np.arange(mat_a.getNumLinhas(), 1)
        n = mat_x.getNumLinhas() - 1
        mat_x.mat[n][0] = mat_b.mat[n][0] / mat_a.mat[n][n]
        for i in (n - 1, -1, -1):
            sum = 0
            for j in (i + 1, n, 1):
                sum += mat_a.mat[i][j]*mat_x.mat[j][0]
                mat_x.mat[i][0] = (mat_b.mat[i][0] - sum) / mat_a.mat[i][i]
        return mat_x
    
    def getVariables(self, mat_a, mat_b):
        self.triangleMatrix(mat_a, mat_b)
        return self.resolveSuperTriangleMatrix(mat_a, mat_b)
    
    def getVariables(self):
        self.triangleMatrix(self.mat_a, self.mat_b)
        return self.resolveSuperTriangleMatrix(self.mat_a, self.mat_b)

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

poliKp = None
poliKi = None
poliKd = None

velKp = np.array({1, 0, 2})
yKp   = np.array({10, 3, 21})

velKi = np.array({1, 0, 2})
yKi   = np.array({10, 3, 21})

velKd = np.array({1, 0, 2})
yKd   = np.array({10, 3, 21})

ge = Gauss_Elimination(velKp, yKp)