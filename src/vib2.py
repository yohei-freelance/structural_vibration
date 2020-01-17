#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 10:21:15 2019

@author: yohei
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 05:24:06 2019

@author: yohei
"""

import numpy as np
import matplotlib.pyplot as plt

E = 70 * (10 ** 9) #youngs_modulus[Pa]
h = 0.002 #thickness[m]
rho = 2700 #desity[kg/m^3]
l = 1 #length[m]
lambda_r = np.array([4.7300407448627,
                     7.8532046240958,
                     10.995607838002,
                     14.137165491258])
beta = np.zeros(4)
for i in range(len(beta)):
    beta[i] += (np.cosh(lambda_r[i])-np.cos(lambda_r[i]))/(np.sinh(lambda_r[i])-np.sin(lambda_r[i]))
        
class BEF: #Both Edge Free  
    def __init__(self):
        self.N = 1000 #division number
        self.x_ = np.linspace(0, l, self.N)
    
    def phi(self, x):
        phi = np.array([np.cosh(lambda_r[i]*x/l) + np.cos(lambda_r[i]*x/l) - beta[i]*(np.sinh(lambda_r[i]*x/l)+np.sin(lambda_r[i]*x/l)) for i in range(4)])
        dif = np.array([(lambda_r[i]/l)**2 for i in range(4)])
        ddphi = np.array([dif[i]*(np.cosh(lambda_r[i]*x/l) - np.cos(lambda_r[i]*x/l) - beta[i]*(np.sinh(lambda_r[i]*x/l)-np.sin(lambda_r[i]*x/l))) for i in range(4)])
        return phi, ddphi
     
    def f(self, x):
        EI = E * h**3 * (1 + 2*x/l)**3 / 12
        phi_for_EI = [[],[],[],[]]
        ddphi = self.phi(x)[1]
        for i in range(4):
            for j in range(4):
                phi_for_EI[i].append(ddphi[i]*ddphi[j]*EI)
        return phi_for_EI
    
    def g(self, x):
        mu = rho*h*(1+2*x/l)
        phi_for_mu = [[],[],[],[]]
        phi = self.phi(x)[0]
        for i in range(4):
            for j in range(4):
                phi_for_mu[i].append(phi[i]*phi[j]*mu)
        return phi_for_mu
    
    def K(self):
        K = np.zeros((4,4))
        for i in range(4):
            for j in range(4):
                for k in range(0, len(self.x_)-2, 2):
                    y=(self.f(self.x_[k])[i][j]+4*self.f(self.x_[k+1])[i][j]+4*self.f(self.x_[k+2])[i][j])/3
                    K[i][j]+=(self.x_[k+1]-self.x_[k])*y
        return K
    
    def M(self):
        M = np.zeros((4,4))
        for i in range(0,4):
            for j in range(0,4):
                for k in range(len(self.x_)-1):
                    y=(self.g(self.x_[k])[i][j]+self.g(self.x_[k+1])[i][j])/2
                    M[i][j]+=(self.x_[k+1]-self.x_[k])*y 
        return M
    
    def eigen(self):
        M_inv = np.linalg.inv(self.M())
        dots = np.dot(M_inv, self.K())
        eigenvalue, eigenvector = np.linalg.eig(dots.T)
        return np.sqrt(eigenvalue), eigenvector

    def discretization(self):
        ector = self.eigen()[1]
        y = np.zeros((4,len(self.x_)))
        for k in range(4):
            for i in range(len(self.x_)):
                temp = 0
                n = self.x_[i]
                for j in range(4):
                    temp += ector[k][j]*self.phi(n)[0][j]
                y[k][i] = temp
        return y        

    def describe_tapered(self):
        y = self.discretization()
        plt.title("Free Vibration of Tapered beam by Simpson method")
        plt.plot(self.x_, -y[3], label="mode:1")
        plt.plot(self.x_, -y[0], label="mode:2")
        plt.plot(self.x_, y[1], label="mode:3")
        plt.plot(self.x_, y[2], label="mode:4")
        plt.legend()
        plt.xlabel("l [m]")
        plt.ylabel("u (x) []")
        plt.savefig('fig7.png')
        
    def describe_uniform(self):
        y = self.phi(self.x_)
        plt.title("Free Vibration of Uniform beam")
        for i in range(4):
            plt.plot(self.x_, y[0][i], label="mode:"+ str(i+1))
        plt.legend()
        plt.xlabel("l [m]")
        plt.ylabel("u (x) []")
        plt.savefig('fig8.png')
    
    def describe_mode1(self):
        y1 = self.discretization()
        y2 = self.phi(self.x_)
        plt.title("Free vibration of Mode1 by Simpson method")
        plt.plot(self.x_, -y1[3], label="Tapered")
        plt.plot(self.x_, y2[0][0], label="Uniform")
        plt.legend()
        plt.xlabel("l [m]")
        plt.ylabel("u (x) []")
        plt.savefig('fig9.png')
        
    def describe_mode2(self):
        y1 = self.discretization()
        y2 = self.phi(self.x_)
        plt.title("Free vibration of Mode2 by Simpson method")
        plt.plot(self.x_, -y1[0], label="Tapered")
        plt.plot(self.x_, y2[0][1], label="Uniform")
        plt.legend()
        plt.xlabel("l [m]")
        plt.ylabel("u (x) []")
        plt.savefig('fig10.png')
        
    def describe_mode3(self):
        y1 = self.discretization()
        y2 = self.phi(self.x_)
        plt.title("Free vibration of Mode3 by Simpson method")
        plt.plot(self.x_, y1[1], label="Tapered")
        plt.plot(self.x_, y2[0][2], label="Uniform")
        plt.legend()
        plt.xlabel("l [m]")
        plt.ylabel("u (x) []")
        plt.savefig('fig11.png')
        
    def describe_mode4(self):
        y1 = self.discretization()
        y2 = self.phi(self.x_)
        plt.title("Free vibration of Mode4 by Simpson method")
        plt.plot(self.x_, y1[2], label="Tapered")
        plt.plot(self.x_, y2[0][3], label="Uniform")
        plt.legend()
        plt.xlabel("l [m]")
        plt.ylabel("u (x) []")
        plt.savefig('fig12.png')

def free_vibration():
    root = np.sqrt((E*(h**2))/(3*rho))
    temp = []
    for i in range(len(lambda_r)):
        temp.append(root*(lambda_r[i]**2)/l**2)
    return temp

if __name__ == '__main__':
    bef = BEF()
    #uni = uniform()
    print("Eigenvalue and Eigenvector", bef.eigen())
    print("equivalent to Eigenvalue of free vibration", free_vibration())
    #bef.describe_tapered()
    bef.describe_uniform()
    #bef.describe_mode1()
    #bef.describe_mode2()
    #bef.describe_mode3()
    #bef.describe_mode4()