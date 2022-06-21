#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from multipledispatch import dispatch

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class Matrix:
    def __init__(self):
        self.linhas  = 1
        self.colunas = 1
        self.mat = None

    #@dispatch(np.ndarray)
    # matriz do tipo float[][]
    def __init__(self, m = None):
        self.linhas  = 1
        self.colunas = 1
        self.mat = None
        if(m == None):
            return
        if(m.length > 0):
            self.linhas = m.length
            self.colunas = m[0].length
        self.mat = m

    #@dispatch(np.ndarray)
    # vetor do tipo float[]
    def __init__(self, m):
        self.linhas  = 1
        self.colunas = 1
        self.mat = None
        self.linhas = 1
        self.colunas = m.length
        self.mat = np.array([self.linhas][self.colunas])
        self.mat[0] = m

    #@dispatch(int, int)
    def __init__(self, l, c = -1):
        self.linhas  = 1
        self.colunas = 1
        self.mat = None
        if(c == -1):
            self.mat = np.array([l][l])
            self.linhas = l
            self.colunas = l
            return
        self.mat = np.array([l][c])
        self.linhas = l
        self.colunas = c

    def printMatrix(self):
        #println(linhas, colunas)
        for i in range(self.linhas):
            print("ln")
            for j in range(self.colunas):
                print("{:3.2f}".format(self.mat[i][j]) + " \t")
        print("\ln")

    
    #@dispatch(np.ndarray)
    def printMatrix(self, mat):
        for i in range(mat.length):
            print("\ln")
            for j in range(mat[0].length):
                print("{:3.2f}".format(mat[i][j]) + " \t")
        print("\ln")
    
    #@dispatch(object)
    def printMatrix(self, A):
        for i in range(A.getNumLinhas()):
            print("ln")
            for j in range(A.getNumColunas()):
                print("{:3.2f}".format(A.mat[i][j]) + " \t")
        print("\ln")

    def transposta(self):
        C = Matrix(self.colunas, self.linhas)[0]
        for i in range(self.colunas):
            for j in range(self.linhas):
                C.mat[i][j] = self.mat[j][i]
        self.mat = C.mat.clone()
        self.colunas = self.linhas
        self.linhas  = self.mat.length

    #@dispatch(object)
    def transposta(self, A):
        C = Matrix(A.getNumColunas(), A.getNumLinhas())
        for i in range(A.getNumLinhas()):
            for j in range(A.getNumColunas()):
                C.mat[i][j] = A.mat[j][i]
        C.colunas = C.linhas
        C.linhas  = C.mat.length
        return C

    def copy(self):
        C = Matrix(self.linhas, self.colunas)
        for i in range(self.linhas):
            for j in range(self.colunas):
                C.mat[i][j] = self.mat[i][j]
        return C

    #@dispatch(object)
    def copy(self, A):
        C = Matrix(A.getNumLinhas(), A.getNumColunas())
        for i in range(A.getNumLinhas()):
            for j in range(A.getNumColunas()):
                C.mat[i][j] = A.mat[i][j]
        return C

    #@dispatch(object, object)
    def sum(self, A, B):
        if ((A.getNumColunas() != B.getNumColunas()) or (A.getNumLinhas() != B.getNumLinhas())):
            return A
        C = Matrix(A.getNumColunas(), A.getNumLinhas())
        for i in range(A.getNumLinhas()):
            for j in range(A.getNumColunas()):
                C.mat[i][j] = A.mat[i][j] + B.mat[i][j]
        return C

    #@dispatch(object)
    def sum(self, A):
        if((self.colunas != A.getNumColunas()) or (self.linhas != A.getNumLinhas())):
            return
        for i in range(A.getNumLinhas()):
            for j in range(A.getNumColunas()):
                self.mat[i][j] += A.mat[i][j]

    #@dispatch(object, object)
    def sub(self, A, B):
        if((A.getNumColunas() != B.getNumColunas()) or (A.getNumLinhas() != B.getNumLinhas())):
            return A
        C = Matrix(A.getNumColunas(), A.getNumLinhas())
        for i in range(A.getNumLinhas()):
            for j in range(A.getNumColunas()):
                C.mat[i][j] = A.mat[i][j] - B.mat[i][j]
        return C

    #@dispatch(object)
    def sub(self, A):
        if((self.colunas != A.getNumColunas()) or (self.linhas != A.getNumLinhas())):
            return
        for i in range(A.getNumLinhas()):
            for j in range(A.getNumColunas()):
                self.mat[i][j] += -A.mat[i][j]

    
    #@dispatch(object, object)
    def mult(self, A, p):
        C = Matrix(A.getNumColunas(), A.getNumLinhas())
        for i in range(A.getNumLinhas()):
            for j in range(A.getNumColunas()):
                C.mat[i][j] = A.mat[i][j]*p
        return C

    #@dispatch(object, object)
    def mult(self, A, B):
        if((A.getNumColunas() != B.getNumLinhas())):
            return A
        C = Matrix(A.getNumLinhas(), B.getNumColunas())
        for i in range(A.getNumLinhas()):
            for j in range(B.getNumColunas()):
                C.mat[i][j] = 0
            for k in range(A.getNumColunas()):
                C.mat[i][j] += A.mat[i][k] * B.mat[k][j]
        return C

    #@dispatch(object)
    def mult(self, p):
        for i in range(self.linhas):
            for j in range(self.colunas):
                self.mat[i][j] = self.mat[i][j]*p

    #@dispatch(object)
    def mult(self, A):
        if((self.colunas != A.getNumLinhas())):
            return
        C = Matrix(self.linhas, A.getNumColunas())
        for i in range(self.linhas):
            for j in range(A.getNumColunas()):
                C.mat[i][j] = 0
            for k in range(self.colunas):
                C.mat[i][j] += self.mat[i][k] * A.mat[k][j]
        self.mat = C.mat
        self.linhas = C.linhas
        self.colunas = C.colunas

    def updateValues(self, values):
        if (self.linhas > 1 and self.colunas > 1):
            return
        if (values.length == self.linhas):
            for i in range(self.linhas):
                self.mat[i][0] = values[i]
        if (values.length == self.colunas):
            self.mat[0] = values

    # @dispatch(*int)
    # def updateValues(self, values):
    #     if (self.linhas > 1 and self.colunas > 1):
    #         return
    #     if (values.length == self.linhas):
    #         for i in range(self.linhas):
    #             self.mat[i][0] = values[i]
    #     if (values.length == self.colunas):
    #         for i in range(self.colunas):
    #             self.mat[i][0] = values[i]

    def swapLines(self, l0, l1):
        if (l0 <= 0 or l0 > self.linhas or l1 <= 0 or l1 > self.linhas or l0 == l1):
            return
        li = None
        l0 = l0 - 1
        l1 = l1 - 1
        for i in range(self.colunas):
            li = self.mat[l0][i]
            self.mat[l0][i] = self.mat[l1][i]
            self.mat[l1][i] = li

    def swapColumns(self, c0, c1):
        if (c0 <= 0 or c0 > self.colunas or c1 <= 0 or c1 > self.colunas or c0 == c1):
            return
        col = None
        c0 = c0 - 1
        c1 = c1 - 1
        for i in range(self.linhas):
            col = self.mat[i][c0]
            self.mat[i][c0] = self.mat[i][c1]
            self.mat[i][c1] = col

    def getLine(self, li):
        if(li <= 0 or li > self.linhas):
            return None
        li = li - 1
        lin = Matrix(1, self.colunas)
        for i in range(lin.getNumColunas()):
            lin.mat[0][i] = self.mat[li][i]
        return lin

    def getColunm(self, col):
        if(col <= 0 or col > self.linhas):
            return None
        col = col - 1
        colunm = Matrix(self.linhas, 1)
        for i in range(colunm.getNumLinhas()):
            colunm.mat[i][0] = self.mat[i][col]
        return colunm

    def getNumLinhas(self):
        return self.linhas

    def getNumColunas(self):
        return self.colunas

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class Gauss_Elimination:
    def __init__(self, mat_a, mat_b):
        self.mat_a = None
        self.mat_x = None
        self.mat_b = None
        if(mat_a.getNumColunas() != mat_a.getNumLinhas() or mat_a.getNumColunas() == 0):
            return
        if(mat_b.getNumLinhas() == 1):
            mat_b.transposta()
        self.mat_a = mat_a.copy()
        self.mat_x = Matrix(mat_a.getNumLinhas(), 1)
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
        mat_x = Matrix(mat_a.getNumLinhas(), 1)
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

class Least_Square:
    
    def __init__(self, order, x, y):
        self.order = 1
        self.mat = None
        self.b_vet = None
        self.sumX = None
        self.sumXY = None
        self.x_values = None
        self.y_values = None
        if(x.shape == None):
            print("Error: More data is needed\n")
            return
        if(x.shape[0] != y.shape[0]):
            print("Error: different size vectors\n")
            return
        self.order = order
        self.x_values = x
        self.y_values = y
        self.mat = Matrix(self.order + 1)
        self.b_vet = Matrix(self.order + 1, 1)
        self.sumX = np.array[2*self.order + 1]
        self.sumXY = np.array[self.order + 1]
  
    def summation(self):
        for i in range(self.order*2):
            for j in range(self.x_values.length):
                sum = 1
                for k in range(i):
                    sum *= self.x_values[j]
                    self.sumX[i] += sum
        for i in range(self.order + 1):
            for j in range(self.y_values.length):
                sum = 1
                for k in range(k < i):
                    sum *= self.x_values[j]
                    self.sumXY[i] += sum*self.y_values[j]
        for i in range(self.mat.linhas):
            for j in range(self.mat.colunas):
                self.mat.mat[i][j] = self.sumX[i + j]
        for j in range(self.b_vet.linhas):
            self.b_vet.mat[j][0] = self.sumXY[j]

  
    def resolve(self):
        self.summation()
        res = Gauss_Elimination(self.mat, self.b_vet)
        return res.getVariables()
    
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

def printFuncao(m):
    print("f(x) = ") # + nf(m.mat[0][0], 1, 2) + " ")
    for i in range(m.getNumLinhas()):
        print("+ (" + "{:10.2f}".format(m.mat[i][0]) + ")x^" + i + " ")
    print("\n")

def printFuncaoArduino(self, m, text):
  print("float get" + text + "(float vel) {\n")
  print("  return " + "{:10.2f}".format(m.mat[0][0]) + " ")
  for i in range(m.getNumLinhas()):
      print("+ (" + "{:10.2f}".format(m.mat[i][0]) + ")*pow(vel, " + i + ") ")
  print(";\n}\n")

def setup():
  velKi = velKd = velKp
  poliKp = Least_Square(2, velKp, yKp)
  poliKi = Least_Square(2, velKi, yKi)
  poliKd = Least_Square(2, velKd, yKd)
  rsKp = poliKp.resolve()
  rsKi = poliKi.resolve()
  rsKd = poliKd.resolve()
  printFuncaoArduino(rsKp, "Kp")
  print("\n")
  printFuncaoArduino(rsKi, "Ki")
  print("\n")
  printFuncaoArduino(rsKd, "Kd")
  print("\n")

def draw():
    a = None

setup()